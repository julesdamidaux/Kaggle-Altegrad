"""Complete Graph-to-Text model combining Graph Encoder, Q-Former, and LLM."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from .graph_encoder import PretrainedGraphormerEncoder
from .qformer import QFormer
import config


class GraphToTextModel(nn.Module):
    """
    Complete model for molecular graph captioning:
    1. Graph Encoder: Pretrained Graphormer (FROZEN)
    2. Q-Former: Extracts relevant features via learnable queries
    3. LLM Decoder: Generates text description
    """
    
    def __init__(self, freeze_llm=False, token=None):
        super().__init__()
        
        print(f"Loading pretrained Graphormer: {config.GRAPHORMER_MODEL_NAME}")
        self.graph_encoder = PretrainedGraphormerEncoder(
            model_name=config.GRAPHORMER_MODEL_NAME,
            hidden_dim=config.GRAPH_HIDDEN_DIM,
            freeze=config.FREEZE_GRAPH_ENCODER
        )
        
        self.qformer = QFormer(
            num_queries=config.NUM_QUERY_TOKENS,
            hidden_dim=config.QFORMER_HIDDEN_DIM,
            num_layers=config.QFORMER_NUM_LAYERS,
            num_heads=config.QFORMER_NUM_HEADS
        )
        
        print(f"Loading LLM: {config.LLM_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL_NAME,
            trust_remote_code=True,
            padding_side='right',
            token=token
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None,
            token=token
        )
        
        if freeze_llm:
            print("Freezing LLM parameters...")
            for param in self.llm.parameters():
                param.requires_grad = False
        
        llm_hidden_size = self.llm.config.hidden_size
        self.graph_to_llm_proj = nn.Linear(config.QFORMER_HIDDEN_DIM, llm_hidden_size)
        
        # Projection from LLM hidden size to Q-Former hidden dimension for text embeddings
        self.text_proj = nn.Linear(self.llm.config.hidden_size, config.QFORMER_HIDDEN_DIM)
        
        # ------ SHARED PREFIX FOR TRAINING + GENERATION ------
        # Keep the trailing space: this is part of the tokenization.
        self.prefix_text = "The molecule is a "
        
        # Two-shot prompt with medium-length, diverse examples (Acid, Disaccharide, Glucoside)
        self.few_shot_examples = [
            "The molecule is a monochlorobenzoic acid carrying a chloro substituent at position 3. It has a role as a drug metabolite. It derives from a benzoic acid. It is a conjugate acid of a 3-chlorobenzoate.",
            "The molecule is a disaccharide consisting beta-D-glucosyl and D-glucuronic acid residues joined by a (1->3)-linkage. It is a carbohydrate acid and a glycosylglucopyranuronic acid. It derives from a cellobiose. It is a conjugate acid of a 3-O-beta-D-glucosyl-D-glucuronate.",
            "The molecule is a beta-D-glucoside consisting of cis-2-coumaric acid having a beta-D-glucosyl residue attached to the phenolic hydroxy group. It derives from a cis-2-coumaric acid. It is a conjugate acid of a 2-(beta-D-glucosyloxy)-cis-cinnamate."
        ]
        
        # Temperature for ITC loss
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        
        # LM head for ITG (text generation in Q-Former)
        self.lm_head = nn.Linear(config.QFORMER_HIDDEN_DIM, len(self.tokenizer))

    def apply_lora(self):
        """Applies LoRA adapters to the LLM."""
        print("LoRA enabled. Applying Low-Rank Adapters to LLM.")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            target_modules=["q_proj", "v_proj"] # Target attention layers
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
    
    def compute_itc(self, graph_node_features, text_inputs, graph_mask=None):
        """
        Image-Text Contrastive Learning (ITC).
        Aligns graph and text representations using in-batch negatives.
        """
        batch_size = graph_node_features.size(0)
        
        # Get raw LLM text embeddings and project to Q-Former hidden dimension
        text_embeds_raw = self.llm.get_input_embeddings()(text_inputs['input_ids'])
        text_embeds = self.text_proj(text_embeds_raw.to(self.text_proj.weight.dtype))
        text_atts = text_inputs['attention_mask']
        
        # Get graph and text features from Q-Former
        graph_feat, text_feat = self.qformer(
            graph_node_features=graph_node_features,
            text_embeds=text_embeds,
            text_atts=text_atts,
            graph_mask=graph_mask,
            mode='itc'
        )
        
        # Compute similarity matrix
        sim_g2t = graph_feat @ text_feat.t() / self.temp
        sim_t2g = sim_g2t.t()
        
        # Labels: diagonal elements are positives
        targets = torch.arange(batch_size, device=graph_feat.device)
        
        # Cross-entropy loss for both directions
        loss_g2t = nn.functional.cross_entropy(sim_g2t, targets)
        loss_t2g = nn.functional.cross_entropy(sim_t2g, targets)
        
        loss_itc = (loss_g2t + loss_t2g) / 2
        return loss_itc
    
    def compute_itm(self, graph_node_features, text_inputs, graph_mask=None):
        """
        Image-Text Matching (ITM).
        Binary classification: matched or unmatched pairs.
        Uses hard negative mining.
        """
        batch_size = graph_node_features.size(0)
        
        # Get raw LLM text embeddings and project to Q-Former hidden dimension
        text_embeds_raw = self.llm.get_input_embeddings()(text_inputs['input_ids'])
        text_embeds = self.text_proj(text_embeds_raw.to(self.text_proj.weight.dtype))
        text_atts = text_inputs['attention_mask']
        
        # First, compute ITC features to find hard negatives (without computing loss)
        with torch.no_grad():
            graph_feat, text_feat = self.qformer(
                graph_node_features=graph_node_features,
                text_embeds=text_embeds,
                text_atts=text_atts,
                graph_mask=graph_mask,
                mode='itc'
            )
            sim_g2t = graph_feat @ text_feat.t()
            # Select hard negatives
            weights_g2t = nn.functional.softmax(sim_g2t, dim=1)
            weights_g2t.fill_diagonal_(0)  # Mask out positives
            
        # Create matched pairs (positive examples)
        itm_logits_pos = self.qformer(
            graph_node_features=graph_node_features,
            text_embeds=text_embeds,
            text_atts=text_atts,
            graph_mask=graph_mask,
            mode='itm'
        )
        
        # Create mismatched pairs (hard negatives)
        neg_idx = torch.multinomial(weights_g2t, 1).squeeze(1)
        text_embeds_neg = text_embeds[neg_idx]
        text_atts_neg = text_atts[neg_idx]
        
        itm_logits_neg = self.qformer(
            graph_node_features=graph_node_features,
            text_embeds=text_embeds_neg,
            text_atts=text_atts_neg,
            graph_mask=graph_mask,
            mode='itm'
        )
        
        # Concatenate positive and negative examples
        itm_logits = torch.cat([itm_logits_pos, itm_logits_neg], dim=0)
        itm_labels = torch.cat([
            torch.ones(batch_size, dtype=torch.long),   # Positive
            torch.zeros(batch_size, dtype=torch.long)   # Negative
        ], dim=0).to(itm_logits.device)
        
        loss_itm = nn.functional.cross_entropy(itm_logits, itm_labels)
        return loss_itm
    
    def compute_itg(self, graph_node_features, text_inputs, graph_mask=None):
        """
        Image-Grounded Text Generation (ITG).
        Generates text conditioned on graph features.
        Forces queries to extract language-relevant visual features.
        """
        # Get raw LLM text embeddings and project to Q-Former hidden dimension
        text_embeds_raw = self.llm.get_input_embeddings()(text_inputs['input_ids'])
        text_embeds = self.text_proj(text_embeds_raw.to(self.text_proj.weight.dtype))
        text_atts = text_inputs['attention_mask']
        
        # Get text features conditioned on graph with multimodal causal mask (inside Q-Former)
        text_output = self.qformer(
            graph_node_features=graph_node_features,
            text_embeds=text_embeds,
            text_atts=text_atts,
            graph_mask=graph_mask,
            mode='itg'
        )

        # Compute language modeling loss
        lm_logits = self.lm_head(text_output)
        
        # Shift for next-token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = text_inputs['input_ids'][:, 1:].contiguous()
        
        # Mask padding tokens
        shift_labels = shift_labels.masked_fill(
            text_inputs['attention_mask'][:, 1:] == 0, -100
        )
        
        loss_itg = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return loss_itg
    
    def forward_stage1(self, batch):
        """
        Stage 1 forward pass with multi-objective training.
        Computes ITC + ITM + ITG losses.
        """
        graph = batch['graph']
        text_inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        
        # Encode graph
        node_features_dense, _, graph_mask = self.graph_encoder(graph)
        
        # Compute three losses
        loss_itc = self.compute_itc(node_features_dense, text_inputs, graph_mask)
        loss_itm = self.compute_itm(node_features_dense, text_inputs, graph_mask)
        loss_itg = self.compute_itg(node_features_dense, text_inputs, graph_mask)
        
        # Total loss (equal weighting as in BLIP-2)
        loss = loss_itc + loss_itm + loss_itg
        
        return {
            'loss': loss,
            'loss_itc': loss_itc.detach(),
            'loss_itm': loss_itm.detach(),
            'loss_itg': loss_itg.detach()
        }
    
    def get_few_shot_prompt(self, num_examples=2):
        """
        Generate 2-shot prompt.
        """
        prompt = "Generate a detailed chemical description following this pattern:\n\n"
        
        for i, example in enumerate(self.few_shot_examples, 1):
            prompt += f"Example {i}: {example}\n\n"
            
        prompt += "Target Molecule:\n" + self.prefix_text
        return prompt
        
    def forward(self, batch, labels=None, stage=2):
        """
        Forward pass. Supports both Stage 1 (multi-objective) and Stage 2 (LLM) training.
        """
        if stage == 1:
            return self.forward_stage1(batch)
        
        # Stage 2: LLM training
        graph = batch['graph']
        
        # 1. Encode graph
        node_features_dense, _, graph_mask = self.graph_encoder(graph)  # [B, N_nodes, hidden_dim]
        
        # 2. Q-Former (extract mode)
        query_output = self.qformer(
            graph_node_features=node_features_dense,
            graph_mask=graph_mask,
            mode='extract'
        )  # [B, num_queries, hidden_dim]
        
        # 3. Project to LLM dimension
        graph_embeds = self.graph_to_llm_proj(query_output)  # [B, num_queries, llm_hidden_size]
        graph_embeds = graph_embeds.to(self.llm.dtype)
        
        batch_size = graph_embeds.size(0)
        
        if labels is not None:
            input_ids = batch['input_ids']          # [B, T]
            attention_mask = batch['attention_mask']  # [B, T]
            
            # Text embeddings
            text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, T, d]
            
            # Concatenate graph + text embeddings
            inputs_embeds = torch.cat([graph_embeds, text_embeds], dim=1)  # [B, G+T, d]
            
            # Attention mask for full sequence
            graph_attention = torch.ones(
                batch_size,
                graph_embeds.size(1),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            full_attention_mask = torch.cat([graph_attention, attention_mask], dim=1)  # [B, G+T]
            
            # Labels: mask padding tokens
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Concatenate graph labels (masked) + text labels
            graph_labels = torch.full(
                (batch_size, graph_embeds.size(1)),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            
            full_labels = torch.cat([graph_labels, labels], dim=1)  # [B, G+T]

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels,   # HF will internally shift logits vs labels
                return_dict=True
            )
            return outputs.loss
        
        else:
            # Usually you'd return graph_embeds to be used as a prefix for generation
            return graph_embeds

    
    @torch.no_grad()
    def generate(
        self,
        batch,
        max_new_tokens=256,
        num_beams=1,
        temperature=0.3,
        top_p=0.9, 
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        length_penalty=1.0,
        use_few_shot=False,
        min_new_tokens=5
    ):
        """Generate text descriptions for molecular graphs - simplified to match training."""
        graph = batch['graph']
        
        # 1. Encode graph exactly as in training
        node_features_dense, _, graph_mask = self.graph_encoder(graph)
        query_output = self.qformer(
            graph_node_features=node_features_dense,
            graph_mask=graph_mask,
            mode='extract'
        )
        graph_embeds = self.graph_to_llm_proj(query_output)
        graph_embeds = graph_embeds.to(self.llm.dtype)
        
        batch_size = graph_embeds.size(0)
        
        # 2. Start prompt: exactly the same prefix as training
        if use_few_shot:
            start_text = self.get_few_shot_prompt()
        else:
            # Add instruction to guide structure without examples
            # instruction = "Generate a comprehensive and detailed description of the molecule. Include structure, role, derivation, and conjugate status. Do not forget anything.\n"
            instruction = ""
            start_text = instruction + self.prefix_text
            
        start_ids = self.tokenizer(
            [start_text] * batch_size,
            return_tensors='pt',
            padding=True,
            add_special_tokens=True
        ).input_ids.to(graph_embeds.device)
        
        # 3. Build inputs exactly as in training: graph + start text (no separator)
        start_embeds = self.llm.get_input_embeddings()(start_ids)
        inputs_embeds = torch.cat([graph_embeds, start_embeds], dim=1)
        
        # 4. Create attention mask
        attention_mask = torch.ones(
            batch_size, 
            inputs_embeds.size(1), 
            device=inputs_embeds.device,
            dtype=torch.long
        )
        
        # 5. Generate with improved parameters
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': min_new_tokens,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'repetition_penalty': repetition_penalty,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'length_penalty': length_penalty,
        }
        
        # Configure sampling vs beam search
        # Configure sampling vs beam search
        if num_beams > 1:
            # Beam Search (Deterministic)
            generation_kwargs.update({
                'do_sample': False,
                'num_beams': num_beams,
                'early_stopping': True,
            })
        elif temperature > 0:
            # Sampling
            generation_kwargs.update({
                'do_sample': True,
                'temperature': temperature,
                'top_p': top_p,
                'num_beams': 1,
            })
        else:
            # Greedy Search
            generation_kwargs.update({
                'do_sample': False,
                'num_beams': 1,
            })
        
        outputs = self.llm.generate(**generation_kwargs)
        
        # 6. Decode - the output will include our start text
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 7. Clean up (remove </think> tags if present)
        cleaned_texts = []
        for text in generated_texts:
            # Remove thinking tags
            text = text.split('</think>')[-1].strip()
            
            # Clean up instructions if present
            if "detailed description" in text:
                text = text.split("The molecule is a")[-1]
                text = "The molecule is a" + text
            
            # Ensure it starts with our prefix
            if not text.startswith(self.prefix_text.strip()):
                # Sometimes models output "The molecule is a" twice if prompted with it
                if "The molecule is a" in text:
                     idx = text.find("The molecule is a")
                     text = text[idx:]
                else:
                    text = self.prefix_text + text
            
            cleaned_texts.append(text)
            
        return cleaned_texts
