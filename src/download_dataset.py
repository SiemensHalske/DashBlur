import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor

visited_urls = set()
image_urls = set()


def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def is_same_domain(url1, url2):
    return urlparse(url1).netloc == urlparse(url2).netloc


def gather_image_urls(start_url, current_url):
    if current_url in visited_urls:
        return

    visited_urls.add(current_url)

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(current_url, headers=headers, timeout=10)
        r.raise_for_status()

        soup = BeautifulSoup(r.content, 'html.parser')

        img_tags = soup.find_all('img')
        for img_tag in img_tags:
            img_url = urljoin(current_url, img_tag.get('src', ''))
            if is_valid(img_url) and is_same_domain(start_url, img_url):
                image_urls.add(img_url)
                print(f"Image URL listed: {img_url}")

        a_tags = soup.find_all('a')
        for a_tag in a_tags:
            new_url = urljoin(current_url, a_tag.get('href', ''))
            if is_valid(new_url) and is_same_domain(start_url, new_url):
                gather_image_urls(start_url, new_url)

    except (
        requests.exceptions.RequestException,
        requests.exceptions.HTTPError,
        requests.exceptions.Timeout,
        requests.exceptions.TooManyRedirects
    ) as e:
        print(f"Skipping {current_url} due to error: {e}")
    except KeyboardInterrupt:
        print("Skipping to download images due to keyboard interrupt")
        return


def download_image(img_url, folder_path='downloaded_images'):
    img_data = requests.get(img_url).content
    img_name = os.path.join(
        folder_path, os.path.basename(urlparse(img_url).path))
    with open(img_name, 'wb') as img_file:
        img_file.write(img_data)
    print(f"Downloaded: {img_name}")  # Notification for every image downloaded


# Replace with the URL you want to start with
starting_url = 'http://www.olavsplates.com/'
gather_image_urls(starting_url, starting_url)

# Create folder if it doesn't exist
if not os.path.isdir('downloaded_images'):
    os.mkdir('downloaded_images')

# Download all gathered images in parallel
with ThreadPoolExecutor() as executor:
    executor.map(download_image, image_urls)
