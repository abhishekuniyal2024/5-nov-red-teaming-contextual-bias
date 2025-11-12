# Contextual Bias Detection in LLM Responses

This project focuses on detecting and analyzing contextual bias in Large Language Model (LLM) responses. It provides tools to quantify how language models might generate different responses based on subtle changes in input prompts, particularly around identity attributes like gender, Race, or intersectional factors.

## Project Overview

The system works by:
1. Sending paired prompts to multiple LLMs (Gemini 2.0 Flash, GPT-3.5 Turbo, and Llama4 Maverick) that differ by a single identity attribute
2. Analyzing the responses for differences in sentiment, semantic meaning, and adjective usage
3. Generating quantitative bias metrics and qualitative analysis
4. Implementing a bias mitigation wrapper specifically for Gemini 2.0 Flash to reduce identified biases

## Key Features

- **Bias Detection**: Identifies contextual bias in LLM responses through multiple metrics
- **Multi-Model Support**: Compatible with multiple LLM providers including:
  - Google Gemini 2.0 Flash (with bias mitigation wrapper)
  - OpenAI GPT-3.5 Turbo
  - Meta Llama4 Maverick
- **Comprehensive Analysis**: Combines quantitative metrics with human-interpretable insights
- **Bias Mitigation**: Includes a wrapper to reduce bias in model outputs
- **Visualization**: Generates visual reports of bias analysis

## Core Files

### Script Execution Order

Follow these steps in order to run the complete bias analysis pipeline:

1. **`generate_llm_responses.py`**
   - First script to execute
   - Sends prompts to all three LLM providers (Gemini 2.0 Flash, GPT-3.5 Turbo, Llama4 Maverick)
   - Stores responses in the `dataset_response/` directory
   - Handles API interactions and response formatting

2. **`analyze_bias_metrics.py`**
   - Processes the collected responses from `dataset_response/`
   - Computes comprehensive bias metrics (sentiment, semantic similarity, adjective analysis)
   - Generates analysis reports and visualizations in `bias_analysis_output/`

3. **`extract_high_bias.py`**
   - Identifies and extracts prompt-response pairs with the highest bias scores
   - Outputs to `bias_analysis_output/high_bias_pairs.csv`
   - Used for focused analysis and training data preparation

4. **`prepare_fine_tune_dataset.py`**
   - Processes the high-bias pairs into structured training data
   - Outputs to `bias_analysis_output/fine_tune_dataset.json`
   - Formats data for model fine-tuning

5. **`bias_wrapper.py`**
   - Applies fairness wrapper specifically to Gemini 2.0 Flash
   - Measures and compares bias metrics before/after wrapper application
   - Saves comparison reports to `wrapper_output/`
   - Generates detailed bias analysis and mitigation reports

6. **`bias_visualizer.py`**
   - Final step in the pipeline
   - Generates visualizations from the wrapper analysis results
   - Creates multiple visualization files in the `wrapper_output/` directory
   - Includes comparative analysis of bias before and after mitigation
   - Produces summary visualizations for different bias metrics

### Configuration and Requirements
- `requirements.txt`: Lists all Python dependencies
- `.env`: Configuration file for API keys (not included in repo)

### Output Directories
- `bias_analysis_output/`: Contains analysis results and metrics
- `wrapper_output/`: Stores output from the bias wrapper analysis
- `dataset/`: Contains the input dataset with prompt pairs
- `dataset_response/`: contain the dataset with the responses from 3 llm.

## Installation

1. Create and activate a virtual environment:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install SpaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   GROQ_API_KEY=your_groq_key
   ```

## Usage

### Complete Pipeline Execution

Run the scripts in the following order:

1. Generate LLM responses:
   ```bash
   python generate_llm_responses.py
   ```

2. Analyze bias metrics:
   ```bash
   python analyze_bias_metrics.py
   ```

3. Extract high-bias examples:
   ```bash
   python extract_high_bias.py
   ```

4. Prepare fine-tuning dataset:
   ```bash
   python prepare_fine_tune_dataset.py
   ```

5. Apply bias wrapper and generate final report:
   ```bash
   python bias_wrapper.py
   ```

6. Apply bias_visualizer to generate interpretable visuals
   ```bash
   bias_visualizer.py
   ```

## Dependencies

Core dependencies (see `requirements.txt` for specific versions):

### Main Dependencies
- Python 3.10+
- pandas
- openai
- google-generativeai
- python-dotenv
- requests
- tqdm

### NLP & ML Libraries
- spacy
- textblob
- sentence-transformers
- torch (PyTorch)
- numpy
- huggingface_hub

### Visualization
- matplotlib
- seaborn

## Output

The system generates output files in three main directories:

### 1. `dataset_response/`
- Contains the raw responses from all three LLMs (Gemini 2.0 Flash, GPT-3.5 Turbo, and Llama4 Maverick)
- Used as input for the analysis pipeline

### 2. `bias_analysis_output/`
Contains four files generated by the analysis scripts:

- `contextual_bias_results.csv`
  - Quantitative metrics for each response pair
  - Sentiment scores, semantic shifts, and bias scores

- `contextual_bias_comparison_with_bias_notes.csv`
  - Qualitative analysis of detected biases
  - Human-readable summaries of bias patterns

- `fine_tune_dataset.json`
  - Processed dataset ready for model fine-tuning
  - Contains high-bias examples and their annotations

- `high_bias_pairs.csv`
  - Subset of prompt-response pairs with the highest bias scores
  - Used for focused analysis and training

### 3. `wrapper_output/`
Contains files generated by `bias_wrapper.py` and `bias_visualizer.py`:

- `pair_wrapper_results.csv`
  - Before/after comparison of bias mitigation
  - Effectiveness metrics for the bias wrapper
  
- `summary_report.txt`
  - Summary of bias mitigation results
  - Key metrics and observations

- **Visualization Files**:
  - `avg_bias_reduction_by_category.png` - Average bias reduction across different categories
  - `bias_before_vs_after.png` - Comparison of bias scores before and after mitigation
  - `bias_mitigation_status.png` - Overview of mitigation success rates
  - `bias_reduction_distribution.png` - Distribution of bias reduction values

## Methodology

The analysis uses multiple techniques to detect bias:

1. **Sentiment Analysis**: Measures emotional tone differences using TextBlob
2. **Semantic Similarity**: Uses sentence-transformers to detect meaning shifts
3. **Adjective Analysis**: Tracks descriptive word choices that might indicate bias
4. **Contextual Bias Score**: Combined metric that weights the above factors
