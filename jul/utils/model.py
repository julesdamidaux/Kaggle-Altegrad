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
        
        # 1. Pretrained Graphormer Encoder (FROZEN)
        print(f"Loading pretrained Graphormer: {config.GRAPHORMER_MODEL_NAME}")
        self.graph_encoder = PretrainedGraphormerEncoder(
            model_name=config.GRAPHORMER_MODEL_NAME,
            hidden_dim=config.GRAPH_HIDDEN_DIM
        )
        
        # 2. Q-Former
        self.qformer = QFormer(
            num_queries=config.NUM_QUERY_TOKENS,
            hidden_dim=config.QFORMER_HIDDEN_DIM,
            num_layers=config.QFORMER_NUM_LAYERS,
            num_heads=config.QFORMER_NUM_HEADS
        )
        
        # 3. Load LLM
        print(f"Loading LLM: {config.LLM_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL_NAME,
            trust_remote_code=True,
            padding_side='right'
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None  # We'll move to device manually
        )
        
        # Apply LoRA if specified
        if config.USE_LORA:
            print("Applying LoRA to LLM...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none"
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
        
        if freeze_llm:
            print("Freezing LLM parameters...")
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # 4. Projection layer: Q-Former output -> LLM input dimension
        llm_hidden_size = self.llm.config.hidden_size
        self.graph_to_llm_proj = nn.Linear(config.QFORMER_HIDDEN_DIM, llm_hidden_size)
        
        # Special tokens for prompting
        self.prompt_template = "Describe the following molecule: "
        
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
        node_features_dense, _ = self.graph_encoder(graph)  # [batch_size, max_num_nodes, hidden_dim]
        
        # 2. Extract features via Q-Former
        query_output = self.qformer(node_features_dense, graph_mask=None)  # [batch_size, num_queries, hidden_dim]
        
        # 3. Project to LLM dimension
        graph_embeds = self.graph_to_llm_proj(query_output)  # [batch_size, num_queries, llm_hidden_size]
        graph_embeds = graph_embeds.to(self.llm.dtype)  # Cast to LLM dtype (e.g. float16)
        
        # 4. Prepare LLM inputs
        batch_size = graph_embeds.size(0)
        
        if labels is not None:
            # Training mode: prepend graph embeddings to text
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Get text embeddings
            text_embeds = self.llm.get_input_embeddings()(input_ids)
            
            # Concatenate graph and text embeddings
            inputs_embeds = torch.cat([graph_embeds, text_embeds], dim=1)
            
            # Update attention mask
            graph_attention = torch.ones(batch_size, graph_embeds.size(1), device=attention_mask.device)
            attention_mask = torch.cat([graph_attention, attention_mask], dim=1)
            
            # Shift labels to account for graph tokens
            # Labels should be [-100] for graph tokens (ignored in loss)
            graph_labels = torch.full(
                (batch_size, graph_embeds.size(1)), 
                -100, 
                dtype=labels.dtype, 
                device=labels.device
            )
            labels = torch.cat([graph_labels, labels], dim=1)
            
            # Forward through LLM
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            return outputs.loss
        else:
            # Inference mode: just return graph embeddings
            return graph_embeds
    
    @torch.no_grad()
    def generate(self, batch, max_length=256, num_beams=4, temperature=0.7, top_p=0.9):
        """
        Generate text descriptions for molecular graphs.
        
        Args:
            batch: Dictionary with 'graph' key
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            List of generated text strings
        """
        graph = batch['graph']
        
        # 1. Encode graph (returns dense features)
        node_features_dense, _ = self.graph_encoder(graph)
        
        # 2. Q-Former
        query_output = self.qformer(node_features_dense, graph_mask=None)
        
        # 3. Project to LLM
        graph_embeds = self.graph_to_llm_proj(query_output)
        graph_embeds = graph_embeds.to(self.llm.dtype)
        
        batch_size = graph_embeds.size(0)
        
        # 4. Prepare prompt
        prompt_ids = self.tokenizer(
            [self.prompt_template] * batch_size,
            return_tensors='pt',
            padding=True
        ).input_ids.to(graph_embeds.device)
        
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        
        # Concatenate graph and prompt embeddings
        inputs_embeds = torch.cat([graph_embeds, prompt_embeds], dim=1)
        
        # Attention mask
        attention_mask = torch.ones(
            batch_size, 
            inputs_embeds.size(1), 
            device=inputs_embeds.device
        )
        
        # 5. Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove prompt from generated text
        generated_texts = [text.replace(self.prompt_template, "").strip() for text in generated_texts]
        
        return generated_texts
