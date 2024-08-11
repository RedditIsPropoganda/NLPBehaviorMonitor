# Reddit Behavioral Analysis System

## Overview

This project implements an advanced behavioral analysis system specifically designed for Reddit. It uses multi-modal learning to analyze user behavior across text content, subreddit interactions, karma scores, and account age. The system can be used for various applications such as content moderation, user engagement analysis, or behavioral research.

## Features

- Multi-modal analysis combining text, subreddit, karma, and account age data
- Custom neural network architecture with BERT-based text encoding
- Reddit-specific feature encoders for subreddit, karma, and account age
- Advanced pattern analysis using clustering, sentiment analysis, and network graphs
- Model interpretation using Integrated Gradients
- Automated data collection using PRAW (Python Reddit API Wrapper)
- Training pipeline with mixed precision and learning rate scheduling
- Integration with Weights & Biases for experiment tracking

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.10+
- PRAW 7.5+
- scikit-learn 0.24+
- NetworkX 2.6+
- Sentence-Transformers 2.1+
- VADER Sentiment 3.3+
- Captum 0.4+
- Weights & Biases 0.12+

For a complete list of dependencies, see `requirements.txt`.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/reddit-behavioral-analysis.git
   cd reddit-behavioral-analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Reddit API credentials:
   - Create a Reddit account and app at https://www.reddit.com/prefs/apps
   - Update the `config` dictionary in `main.py` with your client ID, client secret, and user agent

## Usage

1. Configure the system by editing the `config` dictionary in `main.py`. You can adjust parameters such as the subreddits to analyze, number of posts to fetch, and model hyperparameters.

2. Run the main script:
   ```
   python main.py
   ```

   This will fetch Reddit data, train the model, and perform behavioral analysis.

3. View the results:
   - Training progress and metrics will be logged to Weights & Biases
   - The best model will be saved as `best_reddit_model.pth`
   - Analysis results will be printed to the console and can be further processed or visualized as needed

## Project Structure

- `main.py`: The main script containing the entire pipeline
- `requirements.txt`: List of required Python packages
- `README.md`: This file, containing project documentation

## Model Architecture

The core of the system is the `RedditBehavioralModel` class, which includes:
- A BERT-based text encoder
- Embedding layers for subreddit information
- Linear encoders for karma and account age
- A fusion transformer to combine all features
- A classifier head for final predictions

## Data Processing

The `fetch_reddit_data` function uses PRAW to collect data from specified subreddits. The `RedditDataset` class prepares this data for model input.

## Analysis Techniques

The system employs several analysis techniques:
- Text clustering using Sentence-Transformers and DBSCAN
- Sentiment analysis with VADER
- Emoji usage analysis
- Behavioral graph construction with NetworkX

## Model Interpretation

The `interpret_reddit_model_decision` function uses Integrated Gradients to attribute predictions to input features, providing insights into model decision-making.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Reddit API for providing access to data
- The creators and maintainers of the open-source libraries used in this project
