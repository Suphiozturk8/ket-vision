from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
from PIL import Image, ImageOps
from pyrogram import Client, filters, enums, idle
import asyncio
import os, time
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API information from environment variables
api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")
bot_token = os.getenv("BOT_TOKEN")
version = os.getenv("VERSION")

# Configure Telegram bot
bot = Client(
    name="ket.vision",
    api_id=api_id,
    api_hash=api_hash,
    bot_token=bot_token,
    parse_mode=enums.ParseMode.MARKDOWN,
    skip_updates=True,
)

# Start message


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default prompts
description_prompt = "Describe this image."
question_prompt = "What is happening in this image?"

# Some switches
autoVision = False

# Thread pool executor
executor = ThreadPoolExecutor()

# Load model and tokenizer
hf_logging.set_verbosity_error()  # Suppress logging except errors
model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)


# Function to resize image to 512x512 pixels
def resize_image(image):
    return ImageOps.fit(image, (512, 512), Image.LANCZOS)


# Function to split message length
def split_message(text, max_length=4096):
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


# image recognition function
async def vision(image_path):
    try:
        # Open and resize image
        image = Image.open(image_path)
        resized_image = resize_image(image)

        # Asynchronous processing
        def process_image():
            # Ensure resized_image is a PIL image
            if not isinstance(resized_image, Image.Image):
                raise ValueError("Resized image is not a PIL image.")

            enc_image = model.encode_image(resized_image)
            result = model.answer_question(enc_image, description_prompt, tokenizer)
            return result

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, process_image)
        return result
    except Exception as e:
        logger.error(f"Error in vision function: {e}")
        raise


# Image processing command
@bot.on_message(filters.photo)
async def process_image(bot, message):
    if autoVision:
        download_folder = os.path.join(os.getcwd(), "downloads")
        os.makedirs(download_folder, exist_ok=True)
        image_path = os.path.join(download_folder, "image.jpg")

        # Download photo and get full file path
        file_path = await message.download(file_name=image_path)
        logger.info(f"Image saved to: {file_path}")

        if not file_path:
            await message.reply_text("Failed to download the image.")
            return

        await message.reply_text("Processing image...")
        try:
            result = await vision(file_path)
            for msg in split_message(result):
                await message.reply_text(msg)
        except FileNotFoundError as e:
            await message.reply_text(f"Error: {e}")
        except Exception as e:
            await message.reply_text(f"An unexpected error occurred: {e}")
        finally:
            # Delete the file
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")


# Start command
@bot.on_message(filters.command(["start", "help"]))
async def start_command(bot, message):
    user = message.from_user.mention
    await message.reply_text(
        "`Ket.vision`\nVersion:`0.3`\n\n"
        f"Hi {user}. I'm a Telegram bot that uses the multimodels to describe images.\n\n"
        "**Avaible commands:**\n"
        "`/vision` - Process an image and describe it.\n"
        "`/autovision` - Toggle AutoVision mode.\n\n"
        "**Status:**\n"
        f"AutoVision mode: `{'enabled' if autoVision else 'disabled'}`\n\n"
        f"**Model:** `{model_id}`\n"
    )

# /autovision command
@bot.on_message(filters.command(["autovision"]))
async def autovision_command(bot, message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    global autoVision
    if len(message.command) > 1 and message.command[1].lower() in ["on", "off"]:
        if message.command[1].lower() == "on":
            autoVision = True
            await message.reply_text(
                "Autovision enabled for this chat. I will automatically describe images."
            )
        elif message.command[1].lower() == "off":
            autoVision = False
            await message.reply_text(
                "Autovision disabled for this chat. I will not automatically describe images."
            )
    else:
        await message.reply_text("Usage: `/autovision` [on/off]")

# Ping command
@bot.on_message(filters.command(["ping"]))
async def ping_command(bot, message):
    start_time = time.time()  # Start time before sending reply
    reply = await message.reply_text("Calculating latency...")
    end_time = time.time()  # End time after sending reply
    
    # Calculate latency in milliseconds
    latency = (end_time - start_time) * 1000
    
    # Update the message with latency
    await reply.edit_text(f"**Latency:** `{latency:.2f}`ms")

# /vision command (get replied image and describe it)
@bot.on_message(filters.command(["vision"]))
async def vision_command(bot, message):
    if not message.reply_to_message:
        await message.reply_text("Reply to an image to describe it.")
        return

    if message.reply_to_message.photo:
        await message.reply_text("Processing image...")
        try:
            image_path = await message.reply_to_message.download()
            result = await vision(image_path)
            for msg in split_message(result):
                await message.reply_text(msg)
        except FileNotFoundError as e:
            await message.reply_text(f"Error: {e}")
        except Exception as e:
            await message.reply_text(f"An unexpected error occurred: {e}")
        finally:
            # Delete the file
            if os.path.exists(image_path):
                os.remove(image_path)
                logger.info(f"Deleted file: {image_path}")
    else:
        await message.reply_text("Reply to an image to describe it.")

# Main function
async def main():
    await bot.start()
    logger.info("Bot started")
    await idle()
    await bot.stop()


if __name__ == "__main__":
    bot.run(main())
