import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import VideoClip
from moviepy.audio.AudioClip import CompositeAudioClip
from datetime import datetime, timedelta
import time
import multiprocessing as mp
from functools import partial
import os
import tempfile
from audio_generation import generate_speech

# Create a project-specific tmp directory
PROJECT_TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(PROJECT_TMP_DIR, exist_ok=True)

# Contact information
contact_name = "Toyota"  # Can be changed to any name
contact_initials = "TR"  # Can be changed to match the name

# Get current time for messages
current_time = datetime.now()
today_str = "Today"
if current_time.hour < 12:
    time_str = f"{current_time.hour}:{current_time.minute:02d} AM"
else:
    hour = current_time.hour if current_time.hour <= 12 else current_time.hour - 12
    time_str = f"{hour}:{current_time.minute:02d} PM"

# Messages with timestamps (timestamps will be hidden except for the first message)
messages = [
    ("Give me one moment, I will call you right back!", "", False),
    ("Wure", "", True),
    ("Sure", "", True),
    ("Hey! Goodmorning this is please call me when free", "", True),
    ("Of course! Give me one moment", "", False),
]

# Parameters
width = 1080
height = 1920
background_color = (241, 241, 246)  # iOS chat background
bubble_colors = [(0, 122, 255), (229, 229, 234)]  # iOS blue and gray
text_colors = [(255, 255, 255), (0, 0, 0)]  # White text for blue bubbles, black for gray

# Cache fonts to improve performance
FONTS = {
    'regular': ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32),
    'small': ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24),
    'header': ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36),
    'status': ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28),
    'icons': ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20),
}

# Try different emoji fonts with fallbacks
try:
    FONTS['emoji'] = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 64)
except OSError:
    try:
        FONTS['emoji'] = ImageFont.truetype("/System/Library/Fonts/AppleColorEmoji.ttf", 64)
    except OSError:
        print("Warning: Emoji font not available, falling back to regular font")
        FONTS['emoji'] = FONTS['regular']

# Update all font references to use cached fonts
font_regular = FONTS['regular']
font_small = FONTS['small']
font_header = FONTS['header']
font_status = FONTS['status']
font_icons = FONTS['icons']
font_emoji = FONTS['emoji']

max_text_width = 600  # Reduced from 700 to keep bubbles narrower
padding = 24  # More padding
spacing = 30  # More spacing between bubbles
top_margin = 300  # Increased from 220 to avoid top UI elements
corner_radius = 25  # Radius for rounded corners
bubble_margin = 60  # Increased from 30 to keep bubbles farther from edges
bottom_margin = 200  # Increased from 120 to avoid bottom UI elements

# Header colors
header_bg = (247, 247, 247)  # Light gray for header
header_border = (209, 209, 214)  # Border color
status_color = (60, 60, 67)  # Dark gray for status text
header_text = (0, 0, 0)  # Black for header text
header_blue = (0, 122, 255)  # iOS blue

# Status bar icons (emoji approximations)
status_icons = {
    'signal': '•••',  # Three dots for signal strength
    'wifi': '⌃',      # Upward chevron for wifi
    'battery': '▮'    # Block for battery
}

# Dummy draw for text sizing
dummy_img = Image.new('RGB', (1, 1))
draw_dummy = ImageDraw.Draw(dummy_img)

def draw_rounded_rectangle(draw, coords, radius, color, border_color=None, border_width=0):
    x1, y1, x2, y2 = coords
    
    if color:
        # Fill
        draw.rectangle([x1+radius, y1, x2-radius, y2], fill=color)
        draw.rectangle([x1, y1+radius, x2, y2-radius], fill=color)
        draw.pieslice([x1, y1, x1+radius*2, y1+radius*2], 180, 270, fill=color)
        draw.pieslice([x2-radius*2, y1, x2, y1+radius*2], 270, 360, fill=color)
        draw.pieslice([x1, y2-radius*2, x1+radius*2, y2], 90, 180, fill=color)
        draw.pieslice([x2-radius*2, y2-radius*2, x2, y2], 0, 90, fill=color)
    
    if border_color and border_width > 0:
        # Border
        draw.line([x1+radius, y1, x2-radius, y1], fill=border_color, width=border_width)  # Top
        draw.line([x1+radius, y2, x2-radius, y2], fill=border_color, width=border_width)  # Bottom
        draw.line([x1, y1+radius, x1, y2-radius], fill=border_color, width=border_width)  # Left
        draw.line([x2, y1+radius, x2, y2-radius], fill=border_color, width=border_width)  # Right
        
        # Corners
        draw.arc([x1, y1, x1+radius*2, y1+radius*2], 180, 270, fill=border_color, width=border_width)
        draw.arc([x2-radius*2, y1, x2, y1+radius*2], 270, 360, fill=border_color, width=border_width)
        draw.arc([x1, y2-radius*2, x1+radius*2, y2], 90, 180, fill=border_color, width=border_width)
        draw.arc([x2-radius*2, y2-radius*2, x2, y2], 0, 90, fill=border_color, width=border_width)

def draw_status_bar(draw):
    # Status bar background
    draw.rectangle([0, 0, width, 60], fill=header_bg)
    
    # Time
    # time_text = "6:46"
    # time_width = draw_dummy.textlength(time_text, font=font_status)
    # draw.text((width//2 - time_width//2, 15), time_text, font=font_status, fill=header_text)

def generate_initials(name):
    """Generate initials from a name."""
    words = name.split()
    if len(words) >= 2:
        return (words[0][0] + words[-1][0]).upper()
    return words[0][:2].upper()

def draw_header(draw, name=None, initials=None):
    # Use provided name/initials or fall back to defaults
    display_name = name or contact_name
    display_initials = initials or contact_initials
    if not display_initials:
        display_initials = generate_initials(display_name)
    
    # Draw status bar
    draw_status_bar(draw)
    
    # Main header background
    draw.rectangle([0, 60, width, top_margin], fill=header_bg)
    draw.line([0, top_margin-1, width, top_margin-1], fill=header_border, width=1)
    
    # Back button with arrow and "244" - adjusted position to respect safe margins
    back_text = "< 244"
    draw.text((bubble_margin, 120), back_text, font=font_header, fill=header_blue)
    
    # Video call icon - using emoji font - adjusted position to respect safe margins
    draw.text((width-bubble_margin-40, 120), "📹", font=font_emoji, fill=None)
    
    # Profile section
    profile_size = 80
    profile_x = (width - profile_size) // 2
    profile_y = 70
    
    # Profile circle
    draw.ellipse([profile_x, profile_y, profile_x + profile_size, profile_y + profile_size], 
                 fill=(158, 158, 158))  # Medium gray for profile
    
    # Profile initials
    initials_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    initials_width = draw_dummy.textlength(display_initials, font=initials_font)
    initials_bbox = initials_font.getbbox(display_initials)
    initials_height = initials_bbox[3] - initials_bbox[1]
    
    draw.text((profile_x + (profile_size - initials_width)//2, 
               profile_y + (profile_size - initials_height)//2), 
              display_initials, font=initials_font, fill=(255, 255, 255))
    
    # Contact name
    name_width = draw_dummy.textlength(display_name, font=font_header)
    draw.text((width//2 - name_width//2, profile_y + profile_size + 15), 
              display_name, font=font_header, fill=header_text)

def draw_input_bar(draw, placeholder_text="Good"):
    # Draw input bar background
    draw.rectangle([0, height-bottom_margin, width, height], fill=header_bg)
    draw.line([0, height-bottom_margin, width, height-bottom_margin], fill=header_border, width=1)
    
    # Draw input field with rounded corners - adjusted to respect safe margins
    field_margin = bubble_margin  # Use the same margin as bubbles for consistency
    field_height = 45  # Reduced height for better iOS look
    field_y = height - bottom_margin + (bottom_margin - field_height) // 2
    field_radius = field_height // 2
    
    # Draw rounded input field
    draw_rounded_rectangle(draw,
                         [field_margin, field_y, width-field_margin, field_y + field_height],
                         field_radius,
                         (255, 255, 255))  # White background
    
    # Add subtle border
    draw_rounded_rectangle(draw,
                         [field_margin, field_y, width-field_margin, field_y + field_height],
                         field_radius,
                         None,
                         border_color=(209, 209, 214),  # iOS border color
                         border_width=1)
    
    # Draw placeholder text
    placeholder_color = (60, 60, 67, 128)  # Semi-transparent gray
    text_x = field_margin + 20  # Adjusted position since no icons
    text_y = field_y + (field_height - font_small.getbbox(placeholder_text)[3]) // 2
    draw.text((text_x, text_y), placeholder_text, 
              font=font_small, fill=placeholder_color)

def split_text(message, font, max_width):
    words = message.split()
    lines = []
    current_line = []
    
    for word in words:
        # Check if word contains emoji
        is_emoji = any(ord(c) > 0x1F300 for c in word)
        test_line = ' '.join(current_line + [word])
        
        # Use appropriate font for measurement
        test_font = font_emoji if is_emoji else font
        w = draw_dummy.textlength(test_line, font=test_font)
        
        if w <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def calculate_message_height(message, timestamp):
    lines = split_text(message, font_regular, max_text_width)
    line_heights = [font_regular.getbbox(line)[3] - font_regular.getbbox(line)[1] for line in lines]
    text_block_height = sum(line_heights) + (len(lines) - 1) * 8
    timestamp_height = font_small.getbbox(timestamp)[3] - font_small.getbbox(timestamp)[1]
    return text_block_height + 2 * padding + spacing + timestamp_height + 10

def find_visible_messages(messages_to_show):
    available_height = height - top_margin - bottom_margin
    total_height = 0
    visible_messages = []
    
    for message in reversed(messages_to_show):
        msg_height = calculate_message_height(message[0], message[1])
        if total_height + msg_height > available_height:
            break
        total_height += msg_height
        visible_messages.insert(0, message)
    
    return visible_messages

def generate_frame(messages_to_show, scroll_progress=1.0, contact_name=None, contact_initials=None, placeholder_text="Good"):
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)
    
    # Draw header and input bar
    draw_header(draw, contact_name, contact_initials)
    draw_input_bar(draw, placeholder_text)
    
    # Optional: Draw safe area rectangle for debugging (uncomment for testing)
    # safe_area_color = (255, 0, 0, 128)  # Semi-transparent red for visibility
    # draw.rectangle([bubble_margin, top_margin, width - bubble_margin, height - bottom_margin], outline=safe_area_color, width=2)
    
    # Create a mask for clipping
    mask = Image.new('L', (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    # Fill the visible message area with white (255)
    mask_draw.rectangle([0, top_margin, width, height - bottom_margin], fill=255)
    
    # Create a temporary image for messages
    msg_img = Image.new('RGB', (width, height), background_color)
    msg_draw = ImageDraw.Draw(msg_img)
    
    # Get visible messages
    visible_messages = find_visible_messages(messages_to_show)
    
    # Handle scrolling
    if scroll_progress < 1.0 and len(messages_to_show) > len(visible_messages):
        visible_messages.insert(0, messages_to_show[-(len(visible_messages) + 1)])
        scroll_y = -calculate_message_height(visible_messages[0][0], visible_messages[0][1]) * scroll_progress
    else:
        scroll_y = 0
    
    y = top_margin + scroll_y
    
    # Draw the day and time for the first visible message
    if visible_messages:
        day_time_text = f"{today_str} {time_str}"
        day_time_width = draw_dummy.textlength(day_time_text, font=font_small)
        msg_draw.text((width//2 - day_time_width//2, y + 10), 
                    day_time_text, font=font_small, fill=status_color)
        y += 50  # Add space after the day/time text
    
    # Draw messages
    for i, (message, timestamp, is_sent) in enumerate(visible_messages):
        lines = split_text(message, font_regular, max_text_width)
        line_heights = []
        for line in lines:
            # Check if line contains emoji
            has_emoji = any(ord(c) > 0x1F300 for c in line)
            font_to_use = font_emoji if has_emoji else font_regular
            bbox = font_to_use.getbbox(line)
            line_heights.append(bbox[3] - bbox[1])
        
        text_block_height = sum(line_heights) + (len(lines) - 1) * 8
        max_line_width = max([draw_dummy.textlength(line, font=font_emoji if any(ord(c) > 0x1F300 for c in line) else font_regular) for line in lines])
        bubble_width = max_line_width + 2 * padding
        bubble_height = text_block_height + 2 * padding
        
        # Position bubbles
        if is_sent:  # Right side (blue bubbles)
            bubble_x = width - bubble_margin - bubble_width
        else:  # Left side (gray bubbles)
            bubble_x = bubble_margin
            
        # Draw the bubble
        draw_rounded_rectangle(msg_draw, 
                             [bubble_x, y, bubble_x + bubble_width, y + bubble_height],
                             corner_radius, 
                             bubble_colors[0] if is_sent else bubble_colors[1])
        
        # Draw message text
        text_y = y + padding
        for line in lines:
            # Check if line contains emoji
            has_emoji = any(ord(c) > 0x1F300 for c in line)
            font_to_use = font_emoji if has_emoji else font_regular
            text_height = font_to_use.getbbox(line)[3] - font_to_use.getbbox(line)[1]
            msg_draw.text((bubble_x + padding, text_y), 
                     line, 
                     font=font_to_use, 
                     fill=text_colors[0] if is_sent else text_colors[1])
            text_y += text_height + 8
        
        y += bubble_height + spacing + 10
    
    # Composite the messages onto the main image using the mask
    img.paste(msg_img, (0, 0), mask)
    
    return np.array(img)

def generate_frame_wrapper(args):
    """Wrapper function for multiprocessing"""
    messages, progress, contact_name, contact_initials, placeholder_text = args
    return generate_frame(messages, progress, contact_name, contact_initials, placeholder_text)

def calculate_message_timing(audio_file, base_delay=0.0):
    """Calculate timing for a message based on its audio duration"""
    try:
        audio = AudioFileClip(audio_file)
        duration = audio.duration
        audio.close()
        return duration  # Remove the base_delay addition to eliminate gaps between messages
    except Exception as e:
        print(f"Warning: Could not get audio duration for {audio_file}: {e}")
        return base_delay  # Return minimal delay if audio duration can't be determined

def strip_emojis(text):
    """Remove emojis and special characters from text for audio generation
    
    This is a more comprehensive approach to remove emojis and special characters
    that might cause issues with text-to-speech engines.
    """
    import re
    
    # Pattern to match emojis and other non-speech characters
    emoji_pattern = re.compile(
        "["
        "\U0001F000-\U0001F9FF"  # Most emojis including symbols & pictographs
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F680-\U0001F6FF"  # Transport & Map symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    
    # Remove emoji characters
    cleaned_text = emoji_pattern.sub(r'', text)
    
    # Remove any leftover unusual characters that might cause TTS issues
    cleaned_text = re.sub(r'[^\w\s,.?!\'"-]', '', cleaned_text)
    
    # Clean up excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # If we've removed everything, return a safe fallback message
    if not cleaned_text or cleaned_text.isspace():
        return "message"
        
    return cleaned_text

def generate_chat_video(
    message_list, 
    placeholder_text="Good", 
    contact_name="Toyota", 
    output_file='chat_video.mp4', 
    speed=1.0,
    generate_audio=False,
    model_path="prince-canuma/Kokoro-82M",
    voice="af_heart"
):
    """
    Generate a chat video from a list of messages using multiprocessing.
    
    Args:
        message_list (list): List of strings representing messages. Even indices are received messages, odd indices are sent messages.
        placeholder_text (str): Text to show in the input field.
        contact_name (str): Name of the contact to show in the header.
        output_file (str): Path to save the output video file.
        speed (float): Animation speed multiplier. Default is 1.0 (normal speed).
                      Higher values make animation faster, lower values make it slower.
        generate_audio (bool): Whether to generate TTS audio for messages. Default is False.
        model_path (str): Path to the TTS model. Only used if generate_audio is True.
        voice (str): Voice ID for TTS. Only used if generate_audio is True.
    
    Returns:
        str: Path to the generated video file.
    """
    # Convert plain messages to the required format (message, timestamp, is_sent)
    formatted_messages = []
    for i, msg in enumerate(message_list):
        is_sent = i % 2 == 1  # Odd indices are sent messages
        formatted_messages.append((msg, "", is_sent))
    
    num_messages = len(formatted_messages)
    fps = 30
    base_hold_frames = 3
    hold_frames = max(1, int(base_hold_frames / speed))

    # Create audio directory in project tmp folder
    temp_dir = os.path.join(PROJECT_TMP_DIR, f"audio_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Created temporary directory for audio: {temp_dir}")

    audio_files = []
    message_timings = []  # When each message should appear
    current_time = 0.0  # Track cumulative time
    
    try:
        # Generate audio files and calculate timings
        if generate_audio:
            print("Generating audio for messages...")
            
            for n, (message, _, is_sent) in enumerate(formatted_messages, 1):
                try:
                    current_voice = "af_heart" if not is_sent else "bm_george"
                    
                    print(f"\nGenerating audio for message {n}/{num_messages}:")
                    print(f"Text: {message}")
                    
                    # Strip emojis before generating audio
                    audio_text = strip_emojis(message)
                    print(f"Text for audio (emojis removed): {audio_text}")
                    
                    audio_file = generate_speech(
                        text=audio_text,
                        model_path=model_path,
                        voice=current_voice,
                        output_dir=temp_dir,
                        file_prefix=f"message_{n}",
                        verbose=True
                    )
                    
                    if audio_file and os.path.exists(audio_file):
                        print(f"Successfully generated audio file: {audio_file}")
                        audio_files.append(audio_file)
                        # Calculate when this message should appear
                        message_timings.append(current_time)
                        # Update timing for next message
                        current_time += calculate_message_timing(audio_file)
                    else:
                        print(f"Warning: Failed to generate audio for message {n}")
                        message_timings.append(current_time)
                        current_time += 1.0  # Default duration if audio fails
                except Exception as e:
                    print(f"Warning: Error generating audio for message {n}: {str(e)}")
                    message_timings.append(current_time)
                    current_time += 1.0  # Default duration if audio fails
        else:
            # Without audio, use fixed timing
            for n in range(num_messages):
                message_timings.append(n * 1.0)  # One second per message
            current_time = num_messages * 1.0

        # Calculate total video duration and frame count
        total_duration = current_time + 1.0  # Add extra second at the end
        total_frames = int(total_duration * fps)

        # Prepare frame generation tasks
        frame_tasks = []
        pop_timings = []
        
        # Generate frames based on message timings
        for frame_num in range(total_frames):
            current_time = frame_num / fps
            
            # Find which messages should be visible at this time
            visible_messages = []
            for i, timing in enumerate(message_timings):
                if current_time >= timing:
                    visible_messages.append(formatted_messages[i])
                    if len(pop_timings) < i + 1:  # If we haven't recorded this pop yet
                        pop_timings.append(timing)
            
            frame_tasks.append((visible_messages, 1.0, contact_name, None, placeholder_text))

        # Generate frames
        num_processes = max(1, mp.cpu_count() - 1)
        print(f"Generating {total_frames} frames using {num_processes} processes...")
        
        with mp.Pool(processes=num_processes) as pool:
            frames = pool.map(generate_frame_wrapper, frame_tasks)

        print("Frame generation complete. Creating video...")

        # Create video clip
        video = ImageSequenceClip(list(frames), fps=fps)

        # Create audio track
        audio_clips = []
        
        # Add pop sounds
        pop_sound_path = os.path.join(os.path.dirname(__file__), "bubble_pop.mp3")
        if os.path.exists(pop_sound_path):
            pop_sound = AudioFileClip(pop_sound_path)
            for pop_time in pop_timings:
                audio_clips.append(pop_sound.with_start(pop_time))
                print(f"Added pop sound at {pop_time:.2f}s")
        
        # Add TTS audio
        if generate_audio and audio_files:
            print("\nAdding TTS audio clips:")
            for i, (audio_file, start_time) in enumerate(zip(audio_files, message_timings)):
                try:
                    if os.path.exists(audio_file):
                        audio_clip = AudioFileClip(audio_file)
                        audio_clips.append(audio_clip.with_start(start_time))
                        print(f"Added audio clip {i+1} at {start_time:.2f}s, duration: {audio_clip.duration:.2f}s")
                except Exception as e:
                    print(f"Warning: Error loading audio file {audio_file}: {str(e)}")

        if audio_clips:
            try:
                print(f"\nCombining {len(audio_clips)} audio clips...")
                final_audio = CompositeAudioClip(audio_clips)
                video.audio = final_audio
                print("Audio combined successfully")
            except Exception as e:
                print(f"Warning: Error combining audio clips: {str(e)}")
                print("Continuing without audio")

        # Write final video
        print("\nWriting video file...")
        video.write_videofile(output_file, fps=fps, audio_codec='aac',
                             codec='libx264', preset='medium',
                             bitrate='8000k')
        
        return output_file
        
    finally:
        # Cleanup temporary audio files
        try:
            if os.path.exists(temp_dir):
                print("\nCleaning up temporary files...")
                for file in os.listdir(temp_dir):
                    try:
                        file_path = os.path.join(temp_dir, file)
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Warning: Failed to delete file {file}: {str(e)}")
                os.rmdir(temp_dir)
                print(f"Deleted temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory {temp_dir}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    messages = [
        "Give me one moment, I will call you right back!",  # Received
        "Sure",  # Sent
        "Hey! Goodmorning please call me when free",  # Sent
        "Of course! Give me one moment"  # Received
    ]
    # Generate video at normal speed with audio
    print("Starting video generation...")
    start_time = time.time()
    generate_chat_video(
        messages,
        "Type a message...",
        "John Doe",
        speed=1.0,  # Normal speed for better synchronization
        generate_audio=True
    )
    end_time = time.time()
    print(f"Video generation completed in {end_time - start_time:.2f} seconds")