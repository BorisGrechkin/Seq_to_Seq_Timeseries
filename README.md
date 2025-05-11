# Sequence to sequence timeseries model

This repository contains an implementation of an LSTM Encoder-Decoder model for sequence-to-sequence time series prediction.

## Prerequisites

- Docker (version 20.10+ recommended)
- NVIDIA Docker runtime (if using GPU acceleration)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BorisGrechkin/Seq_to_Seq_Timeseries.git
   cd Seq_to_Seq_Timeseries

2. **Prepare your data**

3. **Create required directories:**

    ```bash
    mkdir -p Visualization Model_weights Data

4. **Place your input time series files in the Data/ directory**
   
5. **Running the Project**

Build and launch the Docker container with command:

    sh run.sh

This will:

- Build the Docker image
- Start the container with mounted volumes
- Begin model training with default parameters

**Monitor training:**

- Training logs will appear in your terminal
- Output visualizations will be saved to Visualization/
- Model weights will be saved to Model_weights/

**Customizing the Run**

To modify training parameters, edit the config.env script


