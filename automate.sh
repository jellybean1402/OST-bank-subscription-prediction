#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Define directories and file names
DATA_DIR="./data"
MODEL_DIR="./models"
NEW_MODEL="new_model.pkl"
DEPLOYED_MODEL="deployed_model.pkl"

# Check if new data is available in the data directory
echo "Checking for new data..."
if [ -z "$(ls -A $DATA_DIR)" ]; then
    echo "No new data found in $DATA_DIR"
    exit 1
fi

# Run the training script (this will create a new model)
echo "Training new model..."
python train_model.py

# Get the most recently trained model
NEW_MODEL_PATH=$(ls -t $MODEL_DIR/*.pkl | head -n 1)

# If the model is new and performs better, deploy it
if [ -f "$NEW_MODEL_PATH" ]; then
    echo "New model found: $NEW_MODEL_PATH"

    # Deploy the new model (copy it to the deployed model path)
    echo "Deploying new model..."
    cp "$NEW_MODEL_PATH" "$MODEL_DIR/$DEPLOYED_MODEL"

    echo "Stopping any existing Streamlit instances..."
    fuser -k 8501/tcp > /dev/null 2>&1

    # (Optional) Start the app with the deployed model
    echo "Starting the deployment app..."

    # Start Streamlit with fixed port and open access
    streamlit run app.py &
    explorer.exe "http://localhost:8501"

else
    echo "No new model found. Exiting..."
    exit 1
fi
