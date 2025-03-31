import json
from agents.multi_agent_system import MultiAgentSystem
from video_generation import generate_chat_video

def main():
    # Initialize the multi-agent system
    system = MultiAgentSystem()
    
    # Run a query
    query = """
    Technical Deep Dives and Research

        Less is More Reasoning (LIMO): @AymericRoucher highlighted Less is More for Reasoning (LIMO), a 32B model fine-tuned with 817 examples that beats o1-preview on math reasoning, suggesting carefully selected examples are more important than sheer quantity for reasoning.
        Diffusion Models without Classifier-Free Guidance: @iScienceLuvr shared a paper on Diffusion Models without Classifier-free Guidance, achieving new SOTA FID on ImageNet 256x256 by directly learning the modified score.
        Scaling Test-Time Compute with Verifier-Based Methods: @iScienceLuvr discussed research proving verifier-based (VB) methods using RL or search are superior to verifier-free (VF) approaches for scaling test-time compute.
        MaskFlow for Long Video Generation: @iScienceLuvr introduced MaskFlow, a chunkwise autoregressive approach to long video generation from CompVis lab, using frame-level masking for efficient and seamless video sequences.
        Intuitive Physics from Self-Supervised Video Pretraining: @arankomatsuzaki presented Meta's research showing intuitive physics understanding emerges from self-supervised pretraining on natural videos, by predicting outcomes in a rep space.
        Reasoning Models and Verifiable Rewards: @cwolferesearch explained that reasoning models like Grok-3 and DeepSeek-R1 are trained with reinforcement learning using verifiable rewards, emphasizing verification in math and coding tasks and the power of RL in learning complex reasoning.
        NSA: Hardware-Aligned Sparse Attention: @deepseek_ai detailed NSA's core components: dynamic hierarchical sparse strategy, coarse-grained token compression, and fine-grained token selection, optimizing for modern hardware to speed up inference and reduce pre-training costs.



    """
    result = system.run(query)
    
    # Print results
    print("Messages:")
    for msg in result["messages"]:
        print(msg)
    print("\nFinal Content:")
    
    try:
        final_content = result.get("final_content")
        if final_content is None:
            print("No final content received")
            return
            
        if isinstance(final_content, (list, dict)):
            # If it's already a Python object, print directly
            print(json.dumps(final_content, indent=2, ensure_ascii=False))
        elif isinstance(final_content, str):
            # If it's a string, try to parse as JSON
            try:
                # Remove the ```json and ``` markers if they exist
                if final_content.startswith('```json'):
                    final_content = final_content[7:]
                if final_content.endswith('```'):
                    final_content = final_content[:-3]
                final_content = final_content.strip()
                
                # Parse JSON while preserving Unicode characters
                parsed_content = json.loads(final_content)
                print(json.dumps(parsed_content, indent=2, ensure_ascii=False))
                final_content = parsed_content  # Use the parsed content for video generation
            except json.JSONDecodeError as e:
                # If not valid JSON, print as plain text
                print(f"Error parsing JSON: {str(e)}")
                print("Raw content:", final_content)
        else:
            print(f"Unexpected content type: {type(final_content)}")
            print(final_content)
            
    except Exception as e:
        print(f"Error processing final content: {str(e)}")
        return

    output_path = generate_chat_video(
        message_list=final_content,
        placeholder_text="Type a message...",
        contact_name="Tracy",
        output_file="output.mp4",
        speed=0.3,
        generate_audio=True
    )

if __name__ == "__main__":
    main() 