import os
import base64
from tqdm import tqdm

IMAGE_DIR = "./images"  # or wherever your images are
fixed = 0
skipped = 0

for filename in tqdm(os.listdir(IMAGE_DIR)):
    if not filename.lower().endswith(".jpg"):
        continue

    filepath = os.path.join(IMAGE_DIR, filename)

    try:
        with open(filepath, "rb") as f:
            content = f.read()

        # Try decoding base64
        try:
            decoded = base64.b64decode(content, validate=True)
        except base64.binascii.Error:
            skipped += 1
            continue

        # Try verifying decoded image with PIL
        from PIL import Image
        from io import BytesIO

        img = Image.open(BytesIO(decoded))
        img.verify()  # Ensure it's a valid image

        # If valid, overwrite the file
        with open(filepath, "wb") as f:
            f.write(decoded)

        fixed += 1

    except Exception as e:
        skipped += 1

print(f"\nFinished. Fixed {fixed} files, skipped {skipped}.")