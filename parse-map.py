import osmium, json
import osmium.osm

# Credit to where credit is due!
# https://lonvia.github.io/geopython17-pyosmium

class OSMHandler(osmium.SimpleHandler):
    def __init__(self):
        super(OSMHandler, self).__init__()
        self.results = []

    def area(self, area: osmium.osm.Area):
        if area.tags.get("landuse") == "military":
            print(".", end="", flush=True)
            for ring in area.outer_rings():
                self.results.append([(p.lat, p.lon) for p in ring])

if __name__ == "__main__":
    h = OSMHandler()
    print("Processing data...")
    h.apply_file("FI-202409120200.osm.pbf", locations=True, idx="flex_mem")
    print()
    print("Writing JSON...")
    with open("varuskunnat.json", "w") as f:
        json.dump(h.results, f)
    print("Finished.")
