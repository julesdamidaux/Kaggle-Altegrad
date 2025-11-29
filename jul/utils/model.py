"""Complete Graph-to-Text model combining Graph Encoder, Q-Former, and LLM."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

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
    
    def __init__(self, freeze_llm=False):
        super().__init__()
        
        print(f"Loading pretrained Graphormer: {config.GRAPHORMER_MODEL_NAME}")
        self.graph_encoder = PretrainedGraphormerEncoder(
            model_name=config.GRAPHORMER_MODEL_NAME,
            hidden_dim=config.GRAPH_HIDDEN_DIM
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
            padding_side='right'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None
        )
        
        # Apply LoRA if enabled (we will freeze/unfreeze parameters in train.py)
        if config.USE_LORA:
            print("Applying LoRA to LLM...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none"
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
        else:
            print("LoRA disabled. LLM will be fully frozen.")
        
        if freeze_llm:
            print("Freezing LLM parameters...")
            for param in self.llm.parameters():
                param.requires_grad = False
        
        llm_hidden_size = self.llm.config.hidden_size
        self.graph_to_llm_proj = nn.Linear(config.QFORMER_HIDDEN_DIM, llm_hidden_size)
        
        self.separator_token = nn.Parameter(torch.randn(1, 1, llm_hidden_size))
        nn.init.normal_(self.separator_token, mean=0.0, std=0.02)
        
        # Two-shot prompt with medium-length, diverse examples (Acid, Disaccharide)
        self.few_shot_examples = [
            "The molecule is a monochlorobenzoic acid carrying a chloro substituent at position 3. It has a role as a drug metabolite. It derives from a benzoic acid. It is a conjugate acid of a 3-chlorobenzoate.",
            "The molecule is a disaccharide consisting beta-D-glucosyl and D-glucuronic acid residues joined by a (1->3)-linkage. It is a carbohydrate acid and a glycosylglucopyranuronic acid. It derives from a cellobiose. It is a conjugate acid of a 3-O-beta-D-glucosyl-D-glucuronate."
        ]
    
    def get_few_shot_prompt(self, num_examples=2):
        """
        Generate 2-shot prompt.
        """
        prompt = "Generate a detailed chemical description following this pattern:\n\n"
        
        for i, example in enumerate(self.few_shot_examples, 1):
            prompt += f"Example {i}: {example}\n\n"
            
        prompt += "Target Molecule:\nThe molecule is a"
        return prompt
        
    def forward(self, batch, labels=None):
        """
        Args:
            batch: Dictionary with:
                - graph: PyG Batch object
                - input_ids: [batch_size, seq_len] (optional, for training)
                - attention_mask: [batch_size, seq_len] (optional)
            labels: [batch_size, seq_len] for training (optional)
        
        Returns:
            If labels provided: loss
            Else: logits
        """
        graph = batch['graph']
        
        # 1. Encode graph (returns dense features already)
        node_features_dense, _, graph_mask = self.graph_encoder(graph)  # [batch_size, max_num_nodes, hidden_dim]
        
        # 2. Extract features via Q-Former
        query_output = self.qformer(node_features_dense, graph_mask=graph_mask)  # [batch_size, num_queries, hidden_dim]
        
        # 3. Project to LLM dimension
        graph_embeds = self.graph_to_llm_proj(query_output)  # [batch_size, num_queries, llm_hidden_size]
        graph_embeds = graph_embeds.to(self.llm.dtype)  # Cast to LLM dtype (e.g. float16)
        
        # 4. Prepare LLM inputs
        batch_size = graph_embeds.size(0)
        
        if labels is not None:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            labels = batch['input_ids'].clone()
            labels[attention_mask == 0] = -100
            
            text_embeds = self.llm.get_input_embeddings()(input_ids)
            separator_embeds = self.separator_token.expand(batch_size, -1, -1).to(self.llm.dtype)
            inputs_embeds = torch.cat([graph_embeds, separator_embeds, text_embeds], dim=1)

            graph_attention = torch.ones(batch_size, graph_embeds.size(1), device=attention_mask.device)
            separator_attention = torch.ones(batch_size, 1, device=attention_mask.device)
            attention_mask = torch.cat([graph_attention, separator_attention, attention_mask], dim=1)
            
            graph_labels = torch.full(
                (batch_size, graph_embeds.size(1) + 1),
                -100, 
                dtype=labels.dtype, 
                device=labels.device
            )
            labels = torch.cat([graph_labels, labels], dim=1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            return outputs.loss
        else:
            return graph_embeds
    
    @torch.no_grad()
    def generate(self, batch, max_new_tokens=256, num_beams=1, temperature=0.3, top_p=0.9, 
                 repetition_penalty=1.0, no_repeat_ngram_size=0, length_penalty=1.0,
                 use_few_shot=False, min_new_tokens=5):
        """Generate text descriptions for molecular graphs using few-shot prompting."""
        graph = batch['graph']
        
        node_features_dense, _, graph_mask = self.graph_encoder(graph)
        query_output = self.qformer(node_features_dense, graph_mask=graph_mask)
        graph_embeds = self.graph_to_llm_proj(query_output)
        graph_embeds = graph_embeds.to(self.llm.dtype)
        
        batch_size = graph_embeds.size(0)
        
        prompt_text = self.get_few_shot_prompt(num_examples=3)
        prompt_ids = self.tokenizer(
            [prompt_text] * batch_size,
            return_tensors='pt',
            padding=True,
            add_special_tokens=False
        ).input_ids.to(graph_embeds.device)
        
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        separator_embeds = self.separator_token.expand(batch_size, -1, -1).to(self.llm.dtype)
        inputs_embeds = torch.cat([graph_embeds, separator_embeds, prompt_embeds], dim=1)
        
        attention_mask = torch.ones(
            batch_size, 
            inputs_embeds.size(1), 
            device=inputs_embeds.device
        )
        
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty
        )
        
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Post-processing: Prepend start phrase and clean up
        cleaned_texts = []
        for text in generated_texts:
            # Remove any </think> tags and content before them if present (though skip_special_tokens might handle some)
            text = text.split('</think>')[-1].strip()
            
            # Prepend forced start if not present (it won't be, as we forced the prompt end)
            if not text.startswith("The molecule is a"):
                text = "The molecule is a " + text
            
            cleaned_texts.append(text)
            
        return cleaned_texts
