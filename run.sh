# bash
docker build -t boris_ai_model .
docker run --gpus all --rm \
  -v $(pwd)/Data:/app/Data \
  -v $(pwd)/Visualization:/app/Visualization \
  -v $(pwd)/Model_weights:/app/Model_weights \
  boris_ai_model