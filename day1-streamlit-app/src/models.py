"""Cached model/tokenizer loaders. Imports are lazy (inside functions) so
page load stays snappy — only the models a given tab needs get pulled in.
"""

import streamlit as st


@st.cache_resource(show_spinner="Loading DistilBERT (attention model)...")
def load_distilbert():
    from transformers import AutoModel, AutoTokenizer

    name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, output_attentions=True)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading GPT-2 tokenizer...")
def load_gpt2_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


@st.cache_resource(show_spinner="Loading LegalBERT tokenizer...")
def load_legalbert_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")


@st.cache_resource(show_spinner="Loading LegalBERT (embedding model)...")
def load_legalbert_model():
    from transformers import AutoModel, AutoTokenizer

    name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading DistilGPT2 (generation model)...")
def load_distilgpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading Qwen2.5-0.5B-Instruct (~1GB, one-time download)...")
def load_qwen_pipeline():
    import torch
    from transformers import pipeline

    device_map = "auto" if torch.cuda.is_available() else None
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device_map=device_map,
    )
