import requests, json, io, random, threading
import pyproj, PIL.Image, shapely
from termcolor import colored

bbox_t = tuple[float, float, float, float]
image_t = PIL.Image.Image

tf_wgs84_fin = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3067")
tf_fin_wgs84 = pyproj.Transformer.from_crs("EPSG:3067", "EPSG:4326")

def load_and_process_bases(filename: str) -> list[tuple[list, bbox_t]]:
    with open(filename, "r") as f:
        bases = json.load(f)
    def get_bbox(coords):
        n, e = zip(*coords)
        min_n, max_n = min(n), max(n)
        min_e, max_e = min(e), max(e)
        return min_n, min_e, max_n, max_e
    return [(base, get_bbox(base)) for base in bases]

def get_image(bbox: bbox_t, coverage_id: str = "ortokuva_vari") -> image_t | None:
    """
    Accepted coverage_ids:
    - ortokuva_vari
    - ortokuva_vaaravari
    - ortokuva_mustavalko (though this seems to always return black images)
    """
    base_url = "https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2"
    fixed_params = "service=WCS&version=2.0.1&request=GetCoverage&format=image/tiff"
    api_key = "[redacted]"
    url = f"{base_url}?{fixed_params}&api-key={api_key}&CoverageID={coverage_id}&SUBSET=E({bbox[0]},{bbox[2]})&SUBSET=N({bbox[1]},{bbox[3]})"
    #print(f"Requested {colored(bbox, 'cyan')} -> {colored(url, 'dark_grey')}. Sending request... ", end="", flush=True)
    req = requests.get(url)
    if req.status_code == 200:
        #print(colored("request successful.", "green"))
        return PIL.Image.open(io.BytesIO(req.content))
    else:
        #print(colored(f"error, got HTTP {req.status_code}.", "red"))
        return None

def wgs84_centered_to_bboxes(center: tuple[float, float], width: float = 500) -> tuple[bbox_t, bbox_t]:
    e, n = tf_wgs84_fin.transform(*center)
    fin_bbox = e - width / 2, n - width / 2, e + width / 2, n + width / 2
    return (
        (*tf_fin_wgs84.transform(fin_bbox[0], fin_bbox[1]), *tf_fin_wgs84.transform(fin_bbox[2], fin_bbox[3])),
        fin_bbox
    )

def get_images_and_percentage(bases: list[tuple[list, bbox_t]]) -> tuple[tuple[image_t | None, image_t | None], float]:
    if random.random() < 0.5:
        bbox = random.choice(bases)[1]
        point = (random.uniform(bbox[0], bbox[2]), random.uniform(bbox[1], bbox[3]))
    else:
        point = (random.uniform(59, 70), random.uniform(20, 32))
    bbox_wgs84, bbox_fin = wgs84_centered_to_bboxes(point)
    images = get_image(bbox_fin, "ortokuva_vari"), get_image(bbox_fin, "ortokuva_vaaravari")
    image_shape = shapely.box(*bbox_wgs84)
    total_area = image_shape.area
    total_overlap = 0
    for coords, _ in bases:
        base_shape = shapely.Polygon(coords)
        inters = shapely.intersection(image_shape, base_shape)
        percent_overlap = inters.area / total_area
        total_overlap += percent_overlap
    return images, min(total_overlap, 1)

def process_images(images: list[image_t]) -> list[image_t] | None:
    def process_image(image):
        if image is None:
            return None
        resized = image.resize((128, 128))
        pixels = resized.getdata()
        extreme = 0
        for pixel in pixels:
            if pixel == (0, 0, 0) or pixel == (255, 255, 255):
                extreme += 1
        return resized if extreme / len(pixels) < 0.2 else None
    images = [process_image(image) for image in images]
    if None in images:
        return None
    return images

def download_sample(filename_prefix: str, bases: list[tuple[list, bbox_t]]) -> bool:
    images, percent = get_images_and_percentage(bases)
    images = process_images(images)
    if images is not None:
        for image, type in zip(images, "AB"):
            filename = f".\\samples\\{filename_prefix}.{type}.{100 * percent:.0f}.png"
            image.save(filename)
        return True
    return False

def downloader(name_prefix: str, num_samples: int, bases: list[tuple[list, bbox_t]], iolock: threading.Lock):
    count = 0
    while count < num_samples:
        images, percent = get_images_and_percentage(bases)
        images = process_images(images)
        if images is not None:
            base_filename = f"{name_prefix}-{count}"
            for image, type in zip(images, "AB"):
                filename = f".\\samples\\{base_filename}.{type}.{100 * percent:.0f}.png"
                image.save(filename)
            with iolock:
                print(f"Saved {colored(f"{base_filename}.{{A, B}}.{100 * percent:.0f}.png", 'cyan')}.")
            count += 1
        else:
            with iolock:
                print(colored("Sample download failed.", "red"))

if __name__ == "__main__":
    bases = load_and_process_bases("base-data.json")
    num_threads = 8
    iolock = threading.Lock()
    print(f"Spawning {num_threads} threads...")
    threads = [threading.Thread(target=downloader, args=(f"{i}", 1024 / num_threads, bases, iolock)) for i in range(num_threads)]
    for thread in threads: thread.start()
    print("Waiting...")
    for thread in threads: thread.join()
    print("Finished.")
