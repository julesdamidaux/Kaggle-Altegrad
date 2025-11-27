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
        # Match the training data format (ChEBI-style descriptions)
        self.prompt_template = "The molecule is a "
        
        # Few-shot examples for improved generation
        self.few_shot_examples = [
            "The molecule is a disaccharide consisting beta-D-glucosyl and D-glucuronic acid residues joined by a (1->3)-linkage. It is a carbohydrate acid and a glycosylglucopyranuronic acid. It derives from a cellobiose. It is a conjugate acid of a 3-O-beta-D-glucosyl-D-glucuronate.",
            "The molecule is a monochlorobenzoic acid carrying a chloro substituent at position 3. It has a role as a drug metabolite. It derives from a benzoic acid. It is a conjugate acid of a 3-chlorobenzoate."
        ]
    
    def get_few_shot_prompt(self, num_examples=2):
        """
        Generate few-shot prompt with example molecule descriptions.
        
        Args:
            num_examples: Number of examples to include (default: 2)
        
        Returns:
            String containing few-shot examples
        """
        examples = self.few_shot_examples[:num_examples]
        prompt = "Here are examples of molecule descriptions:\n\n"
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}: {example}\n\n"
        prompt += "Now describe this molecule:\n"
        prompt += self.prompt_template
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
            # Training mode: prepend graph embeddings to text
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Mask padding tokens in labels
            labels = batch['input_ids'].clone()
            labels[attention_mask == 0] = -100
            
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
    def generate(self, batch, max_new_tokens=256, num_beams=4, temperature=0.7, top_p=0.9, 
                 repetition_penalty=1.0, no_repeat_ngram_size=0, length_penalty=1.0,
                 use_few_shot=False, min_new_tokens=30):
        """
        Generate text descriptions for molecular graphs.
        
        Args:
            batch: Dictionary with 'graph' key
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            no_repeat_ngram_size: Size of n-grams that can only occur once
            length_penalty: Penalty for length (>1.0 encourages longer, <1.0 shorter)
            use_few_shot: Whether to use few-shot prompting with examples
            min_new_tokens: Minimum number of new tokens to generate (prevents truncation)
        
        Returns:
            List of generated text strings
        """
        graph = batch['graph']
        
        # 1. Encode graph (returns dense features)
        node_features_dense, _, graph_mask = self.graph_encoder(graph)
        
        # 2. Q-Former
        query_output = self.qformer(node_features_dense, graph_mask=graph_mask)
        
        # 3. Project to LLM
        graph_embeds = self.graph_to_llm_proj(query_output)
        graph_embeds = graph_embeds.to(self.llm.dtype)
        
        batch_size = graph_embeds.size(0)
        
        # 4. Prepare prompt (with optional few-shot examples)
        if use_few_shot:
            # Use few-shot prompt with examples
            prompt_text = self.get_few_shot_prompt(num_examples=2)
            prompt_ids = self.tokenizer(
                [prompt_text] * batch_size,
                return_tensors='pt',
                padding=True
            ).input_ids.to(graph_embeds.device)
        else:
            # Use simple prompt
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
        # Note: Cannot use sampling (do_sample=True) with beam search (num_beams > 1)
        # Beam search is deterministic and incompatible with temperature/top_p sampling
        if num_beams > 1:
            # Use beam search (deterministic)
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                early_stopping=True,  # Stop when all beams hit EOS
            )
        else:
            # Use sampling (greedy or stochastic)
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Clean up generated text (keep "The molecule is a" prefix as per training data)
        if use_few_shot:
            # Remove few-shot examples but keep the final prompt template + generation
            cleaned_texts = []
            for text in generated_texts:
                # Find the last occurrence of the prompt template
                if self.prompt_template in text:
                    # Split at last occurrence and keep prompt + everything after
                    parts = text.rsplit(self.prompt_template, 1)
                    cleaned_texts.append(self.prompt_template + parts[-1].strip())
                else:
                    # If prompt not found, prepend it manually
                    cleaned_texts.append(self.prompt_template + text.strip())
            generated_texts = cleaned_texts
        else:
            # Keep the prompt template in the output (matches training data format)
            # Just ensure it's present at the start
            cleaned_texts = []
            for text in generated_texts:
                if not text.startswith(self.prompt_template):
                    cleaned_texts.append(self.prompt_template + text.strip())
                else:
                    cleaned_texts.append(text.strip())
            generated_texts = cleaned_texts
        
        return generated_texts
