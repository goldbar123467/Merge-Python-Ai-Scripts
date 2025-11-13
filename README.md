# Merge Python AI Scripts  
Custom From-Scratch LLM Training, Merging, and Deployment Toolkit

## Overview
This repository contains a collection of original Python scripts I built from scratch to train, merge, and deploy domain-specialized Large Language Models. The goal of this project is to provide a lightweight, fully transparent pipeline for adapting open-source models using parameter-efficient techniques such as LoRA, custom dataset preparation, and optimized local GPU inference.

These scripts form the backbone of my applied AI projects, including a geopolitical forecasting engine and multiple retrieval-augmented reasoning agents.

---

## Key Features

### 🔹 **From-Scratch LoRA Training**
- Minimal, readable implementation for PEFT-style fine-tuning  
- Customizable hyperparameters, sequence lengths, and adapter configs  
- Designed for small, focused datasets and domain specialization  

### 🔹 **Model Merge Pipeline**
- Merge LoRA adapters into base models using custom weight arithmetic  
- Supports multi-stage merging, selective layer merging, and dry-run modes  
- Compatible with HuggingFace, GGUF, and local inference frameworks  

### 🔹 **Dataset Preparation Tools**
- Utilities for cleaning, transforming, and structuring domain-specific text  
- Automatic formatting for Q/A, dialogue, and instruction datasets  
- Tokenizer alignment for custom vocab and prompt styles  

### 🔹 **Local GPU Inference Optimization**
- Memory-aware execution paths for consumer GPUs (e.g., RTX 20xx series)  
- Quantization-friendly design (Q4, Q5, Q8 GGUF workflows)  
- Custom batch loops for low-latency multi-turn reasoning  

### 🔹 **Retrieval-Augmented Generation (RAG) Utilities**
- Optional ingestion scripts for ChromaDB vector stores  
- Embedding generation and metadata-aware retrieval  
- Hooks for plugging merged models directly into RAG pipelines  

---

## Why This Project Matters
These tools were created during my self-taught transition into AI engineering. I built everything without relying on high-level libraries so I could understand **every layer** of the LLM pipeline: data, training, merging, inference, and retrieval.

This repository represents:
- End-to-end ownership of ML workflows  
- Ability to operate with limited hardware  
- Strong applied understanding of LLM internals  
- Real engineering problem-solving, not just running notebooks  

---

## Repository Structure

