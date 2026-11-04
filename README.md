# Where Will the Duke Go?

Small Windows desktop tool for projecting a straight line through a selected target and the village center in a Clash of Clans screenshot.

## Run

```powershell
python -m pip install -r requirements.txt
python dragon_duke_coordinates.py
```

Open a screenshot or paste a copied image with **Ctrl+V**, then click a defense. The two edge coordinates appear in the right panel. Use **Set center by click** if the village is offset in the screenshot. **Clear target** keeps the image and center.

The route is a geometric projection of the straight-through-center path shown in the supplied route reference; the app does not simulate combat. Intersections land on the image rectangle, so crop screenshots to the playable village bounds before using them as deployment-edge coordinates.

## Headless check

Pass any local image to run the headless geometry and rendering check:

```powershell
python dragon_duke_coordinates.py --self-test path\to\base-image.png
```

## Geometry

The app uses `P(t) = C + t(T - C)`. It intersects that line with the four pixel-coordinate edges, skips zero direction components, filters intersections to the image rectangle, and deduplicates corner hits. Coordinates use pixel centers `0..width-1` and `0..height-1`; the automatic center is exactly `(width/2, height/2)`.

The route is a geometric projection of the straight-through-center path. It does not simulate combat. Crop screenshots to the playable village bounds before using the image-edge points as deployment coordinates.

## Build a Windows executable

```powershell
python -m pip install pyinstaller
pyinstaller --noconfirm --clean --onefile --windowed --name WhereWillTheDukeGo dragon_duke_coordinates.py
```

