import requests
from PIL import Image
from io import BytesIO
import json
import time
import argparse



def scrape_image(url,
                 save=True,
                 path='data/misviz/img/test_img.png',
                 timeout=2):
    try:
        #Define user agent for requests
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url,headers=headers,timeout=timeout)
        response.raise_for_status()
        # Open the image from the HTTP response content
        try:
            image = Image.open(BytesIO(response.content))
        except:
            #BitesIO decoding does not work
            print("BytesIO decoding error")
            return 'Error failed to retrieve image'
        if  save:
            try:
                image.save(path)
            except:
                image.convert('RGB').save(path)
        else:
            print(image)
        return 'Successful'

    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve the image: {e}")
        return 'Request exception %s'%e
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_wayback', type=int, default=0, help="If set to 1, use the wayback machine urls to scrape the images")
    parser.add_argument('--sleep',type=int, default=2, help='Time between scraping two images.')
    args = parser.parse_args()

    misviz = json.load(open(f"data/misviz/misviz.json", encoding="utf-8"))
    for m in range(len(misviz)):
        image_path = f"data/misviz/{misviz[m]['image_path']}"
        image_url = misviz[m]['image_url'] if not args.use_wayback else misviz[m]['wayback_image_url']
        scrape_image(image_url, True, path=image_path)
        time.sleep(args.sleep)
    