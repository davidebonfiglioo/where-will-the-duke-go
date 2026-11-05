# Where Will the Duke Go?

Small Windows desktop tool for projecting a straight line through a selected target and the village center in a Clash of Clans screenshot.

Download the latest release — a single `WhereWillTheDukeGo.exe`, so there is nothing to install. This repository holds the full source; see Running from source to build or run it yourself.

## What it does

Open a screenshot or paste a copied image with **Ctrl+V**, then click a defense. The two edge coordinates appear in the right panel. Use **Set center by click** if the village is offset in the screenshot. **Clear target** keeps the image and center.

The route is a geometric projection of the straight-through-center path shown in the supplied route reference; the app does not simulate combat. Intersections land on the image rectangle, so crop screenshots to the playable village bounds before using them as deployment-edge coordinates.

## Geometry

The app uses `P(t) = C + t(T - C)`. It intersects that line with the four pixel-coordinate edges, skips zero direction components, filters intersections to the image rectangle, and deduplicates corner hits. Coordinates use pixel centers `0..width-1` and `0..height-1`; the automatic center is exactly `(width/2, height/2)`.

## Running from source

To run the Python script directly:

```powershell
python -m pip install -r requirements.txt
python dragon_duke_coordinates.py
