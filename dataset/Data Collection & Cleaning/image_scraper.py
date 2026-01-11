import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from PIL import Image
import io

# Website URL to scrape images from
website_url = "https://www.famousfix.com/list/malaysian-male-actors"

# Set up Chrome options
options = Options()
options.add_argument("--start-maximized")  # Open browser in full screen
options.add_argument("--log-level=3")  # Suppress logs
options.add_argument("--disable-blink-features=AutomationControlled")  # Hide automation detection

chrome_driver_path = "./chromedriver.exe"  # Update with your actual path

# Start WebDriver
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

# Open the target website
driver.get(website_url)
time.sleep(5)  # Allow some time for the page to load

# Scroll down to load more images (if the site has lazy loading)
for _ in range(5):  # Adjust based on the website behavior
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

# Find image elements
image_elements = driver.find_elements(By.TAG_NAME, "img")

# Create folder to save images
download_folder = "scraped_images_4"
os.makedirs(download_folder, exist_ok=True)

# Download images
image_urls = set()
for i, img in enumerate(image_elements):
    try:
        img_url = img.get_attribute("src")
        if img_url and "http" in img_url:
            image_urls.add(img_url)
            print(f"Found image: {img_url}")

    except Exception as e:
        print(f"Failed to extract URL: {e}")

print(f"Found {len(image_urls)} unique images. Downloading...")

# Save images
for i, img_url in enumerate(image_urls):
    try:
        response = requests.get(img_url, timeout=5)
        response.raise_for_status()
        image_data = Image.open(io.BytesIO(response.content))

        # Save image
        image_path = os.path.join(download_folder, f"image_{i}.jpg")
        image_data.save(image_path, "JPEG")
        print(f"Saved: {image_path}")

    except Exception as e:
        print(f"Failed to download {img_url}: {e}")

# Close WebDriver
driver.quit()
print(f"Successfully downloaded {len(image_urls)} images!")