# AI Content to Chat Video

A Streamlit application that transforms AI-related content into engaging chat conversations and animated videos.

## Features

- **Research Agent**: Analyzes and summarizes AI-related content
- **Content Creation Agent**: Transforms research into natural chat conversations
- **Video Generation**: Creates animated chat videos with customizable settings
- **User-friendly Interface**: Easy-to-use Streamlit interface with examples and settings

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/create-with-flow.git
cd create-with-flow
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following:
```
GOOGLE_API_KEY=your_google_api_key
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter AI-related content in the text area or choose one of the example topics

4. Click "Generate Content & Video" to process your input

5. View the generated conversation and video

6. Customize video settings and regenerate if desired

7. Download the video when satisfied

## Troubleshooting

### PyTorch Compatibility Issues

If you encounter errors related to PyTorch and event loops, try these solutions:

1. Disable audio generation using the checkbox in the sidebar settings
2. Restart the Streamlit app
3. If the issue persists, try running with a specific Python version (3.9 is recommended)
4. Make sure your environment has compatible versions of PyTorch and Streamlit

Error messages like `RuntimeError: no running event loop` or `Tried to instantiate class '__path__._path'` are typically related to PyTorch and Streamlit compatibility issues. The app includes built-in error handling to help mitigate these issues.

## Configuration

You can configure the following settings:

- **Contact Name**: The name displayed in the chat interface
- **Input Placeholder**: Text shown in the input field of the chat
- **Video Speed**: Playback speed of the generated video

## Examples

The app includes example topics to demonstrate its capabilities:
- GPT-4 Overview
- AI Research Ethics
- Diffusion Models

## Folder Structure

- `agents/`: Contains the multi-agent system implementation
- `video_generation.py`: Handles the creation of chat videos
- `audio_generation.py`: Manages text-to-speech for video audio
- `streamlit_app.py`: Main application file
- `app.py`: Original script for command-line usage

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI API
- MoviePy
- PIL
- Other dependencies listed in requirements.txt

## License

MIT

## Acknowledgements

- This project uses the Google Generative AI API for research and content creation
- Video generation is built with MoviePy and PIL 