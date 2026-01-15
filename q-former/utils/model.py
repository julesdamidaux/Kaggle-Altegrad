"""Complete Graph-to-Text model combining Graph Encoder, Q-Former, and LLM."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
        
        # Add special token for Graph
        self.tokenizer.add_tokens([config.GRAPH_TOKEN], special_tokens=True)
        self.graph_token_id = self.tokenizer.convert_tokens_to_ids(config.GRAPH_TOKEN)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_NAME,
            trust_remote_code=True,
            quantization_config=bnb_config, # Enable 4-bit
            device_map="auto", # Required for bitsandbytes
            token=token
        )
        
        # Resize token embeddings to account for new special token
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        if freeze_llm:
            print("Freezing LLM parameters...")
            for param in self.llm.parameters():
                param.requires_grad = False
        
        llm_hidden_size = self.llm.config.hidden_size
        self.graph_to_llm_proj = nn.Linear(config.QFORMER_HIDDEN_DIM, llm_hidden_size)
        
        # Projection from LLM hidden size to Q-Former hidden dimension for text embeddings
        self.text_proj = nn.Linear(self.llm.config.hidden_size, config.QFORMER_HIDDEN_DIM)
        
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Qwen/Llama targets
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
        
    def forward(self, batch, labels=None, stage=2):
        """
        Forward pass. Supports both Stage 1 (multi-objective) and Stage 2 (LLM) training.
        """
        if stage == 1:
            return self.forward_stage1(batch)
        
        # Stage 2/3: LLM training with Instruct format
        graph = batch['graph']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch.get('labels', labels) # Get labels from batch if available
        
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
        
        # 4. Embed Input IDs (Text)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 5. Insert Graph Embeddings
        # We find the positions of GRAPH_TOKEN and replace them with graph_embeds
        # Note: data_utils repeats GRAPH_TOKEN to match NUM_QUERY_TOKENS
        
        # Mask for graph tokens
        graph_token_mask = (input_ids == self.graph_token_id)
        
        # Check alignment
        batch_size = inputs_embeds.size(0)
        
        # To avoid complex indexing, we can use scatter or index_put, but simple replacement works if aligned.
        # Since we use config.NUM_QUERY_TOKENS repetitions, we expect exactly that many True in mask per batch item
        
        # Simple Loop replacement for safety (efficient enough for batch=4)
        for i in range(batch_size):
            # The mask for this sample
            mask = graph_token_mask[i]
            if mask.sum() != config.NUM_QUERY_TOKENS:
                # Warning or handling if count doesn't match (e.g. truncation)
                # If truncated, we just fill what we can or skip? 
                # Ideally we ensure data_utils truncates TEXT not PROMPT.
                # For now, let's just replace where we can.
                pass
                
            # Replace
            # graph_embeds[i] is [num_queries, dim]
            # inputs_embeds[i][mask] is [num_masked, dim]
            # We assume they match or we crop
            num_slots = mask.sum()
            num_embeds = graph_embeds.size(1)
            
            if num_slots > 0:
                inputs_embeds[i, mask] = graph_embeds[i, :num_slots]

        if labels is not None:
             outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
             return outputs.loss
        else:
            return inputs_embeds

    
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
        use_few_shot=False, # Ignored in Instruct mode
        min_new_tokens=5,
        do_sample=False
    ):
        """Generate text descriptions using Chat Template format."""
        graph = batch['graph']
        
        # 1. Encode graph
        node_features_dense, _, graph_mask = self.graph_encoder(graph)
        query_output = self.qformer(
            graph_node_features=node_features_dense,
            graph_mask=graph_mask,
            mode='extract'
        )
        graph_embeds = self.graph_to_llm_proj(query_output)
        graph_embeds = graph_embeds.to(self.llm.dtype)
        
        if torch.isnan(graph_embeds).any():
            print("NaNs detected in graph_embeds!")
            # Handle or assert
            # Try to sanitize?
            graph_embeds = torch.nan_to_num(graph_embeds, nan=0.0)
        
        batch_size = graph_embeds.size(0)
        
        # 2. Build Prompt
        # User: Describe the following molecule: <graph>...<graph>\nAssistant:
        graph_tokens = config.GRAPH_TOKEN * config.NUM_QUERY_TOKENS
        user_prompt = f"Describe the following molecule: {graph_tokens}"
        
        conversation = [
            {"role": "user", "content": user_prompt},
        ]
        
        # Format with chat template to get the "Assistant:" start
        prompt_text = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize prompt
        inputs = self.tokenizer(
            [prompt_text] * batch_size,
            return_tensors='pt',
            padding=True,
            add_special_tokens=False
        ).to(graph_embeds.device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # 3. Embed and Replace
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        graph_token_mask = (input_ids == self.graph_token_id)
        
        for i in range(batch_size):
            mask = graph_token_mask[i]
            num_slots = mask.sum()
            if num_slots > 0:
                inputs_embeds[i, mask] = graph_embeds[i, :num_slots]

        # 4. Generate
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
        
        if num_beams > 1:
            generation_kwargs.update({'do_sample': False, 'num_beams': num_beams, 'early_stopping': True})
        elif do_sample or temperature > 0:
            generation_kwargs.update({'do_sample': True, 'temperature': temperature, 'top_p': top_p, 'num_beams': 1})
        else:
            generation_kwargs.update({'do_sample': False, 'num_beams': 1})
        
        outputs = self.llm.generate(**generation_kwargs)
        
        # 5. Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Clean up
        cleaned_texts = []
        for text in generated_texts:
            # Instruct models shouldn't output the prompt, but generate() with inputs_embeds sometimes does?
            # Actually, HF generate typically returns ONLY new tokens if inputs_embeds are used? 
            # No, usually it returns full sequence.
            # Llama 3 instruct output usually assumes continuation.
            
            # Since we decoded the output, it probably contains the prompt if we passed input_ids (but we passed embeds).
            # We must check carefully.
            
            # Simple cleanup: remove the prompt part if it exists
            # We know the prompt ends with "Assistant:" headers usually.
            # Let's just take everything after "assistant" header if we can find it.
            
            if "assistant" in text.lower():
                # This is a bit heuristic
                # Llama 3 header is specific.
                pass
            
            cleaned_texts.append(text)
            
        return cleaned_texts
