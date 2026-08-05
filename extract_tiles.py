"""
extract_tiles_v2.py

===================
Offline extraction tool for the Lunar MAE foundation model.

WHAT THIS SCRIPT DOES:
1. Scans a target directory for massive LROC .IMG / .TIF satellite rasters.
2. Uses "windowed reads" to chop these multi-gigabyte images into small 224x224 tiles 
   without crashing your system RAM.
3. Analyzes each tile to drop useless empty space (black borders/shadows).
4. Normalizes the valid lunar surface tiles and packs them into a single, highly 
   compressed .npz array.

WHY THIS IS NECESSARY:
Training a ViT/MAE directly from massive rasters on a cloud GPU causes severe I/O 
bottlenecks. By preprocessing the data into a dense, compact format locally, you 
ensure your Colab GPU spends 100% of its time training the model instead of waiting 
for disk reads.
"""

import os
import glob
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window

def process_single_image(img_path, tile_size, stride, blank_std, current_tile_count, max_tiles):
    """
    Extracts valid 224x224 tiles from a single geospatial raster.
    
    Args:
        img_path (str): Path to the .IMG or .TIF file.
        tile_size (int): The height/width of the extracted tile (e.g., 224).
        stride (int): How far to move the extraction window each step. 
                      If stride < tile_size, tiles will overlap.
        blank_std (float): The standard deviation threshold. Tiles below this 
                           are considered blank/featureless and are dropped.
        current_tile_count (int): How many tiles we've collected globally so far.
        max_tiles (int): The global cap to prevent memory overflow.
        
    Returns:
        list: A list of valid, normalized uint8 tiles from this image.
        int: The number of blank tiles skipped in this image.
    """
    valid_tiles = []
    skipped_count = 0
    
    try:
        # rasterio.open reads the metadata, NOT the whole multi-gigabyte file.
        with rasterio.open(img_path) as src:
            height, width = src.height, src.width
            
            # Loop over the image in a grid pattern
            for y in range(0, height - tile_size + 1, stride):
                for x in range(0, width - tile_size + 1, stride):
                    
                    # Stop immediately if we've hit our global dataset size limit
                    if (current_tile_count + len(valid_tiles)) >= max_tiles:
                        return valid_tiles, skipped_count
                        
                    # ---------------------------------------------------------
                    # STEP 1: The Windowed Read
                    # ---------------------------------------------------------
                    # Instead of loading the 8GB image, we tell rasterio to only 
                    # fetch the specific 224x224 pixel block at coordinates (x, y).
                    window = Window(x, y, tile_size, tile_size)
                    tile = src.read(1, window=window).astype(np.float32)

                    # --- ADD THIS NEW CHECK ---
                    # Catch extreme "No Data" values or infinities and drop the tile 
                    # before it causes an overflow in the math below.
                    if not np.all(np.isfinite(tile)) or tile.min() < -10000:
                        skipped_count += 1
                        continue
                    # --------------------------
                    
                    # ---------------------------------------------------------
                    # STEP 2: The Blank Space Filter
                    # ---------------------------------------------------------
                    # NAC ROI images have massive black borders (no-data regions).
                    # A completely black (or flat gray) tile has a standard deviation 
                    # near 0. Training an MAE on blank tiles wastes compute and ruins 
                    # the loss curve. We drop them here.
                    if tile.std() <= blank_std:
                        skipped_count += 1
                        continue
                        
                    # ---------------------------------------------------------
                    # STEP 3: Per-Tile Min-Max Normalization
                    # ---------------------------------------------------------
                    # The moon's absolute brightness changes drastically based on the 
                    # sun angle. We want our model to learn *texture and craters*, 
                    # not absolute lighting. By stretching every individual tile's 
                    # pixels to span exactly 0.0 to 1.0, we force the model to focus 
                    # on structural contrast.
                    tmin, tmax = tile.min(), tile.max()
                    if tmax > tmin:
                        tile_norm = (tile - tmin) / (tmax - tmin)
                    else:
                        skipped_count += 1
                        continue 
                        
                    # ---------------------------------------------------------
                    # STEP 4: Memory Compression (float32 -> uint8)
                    # ---------------------------------------------------------
                    # A 50,000-tile dataset in float32 takes ~10 GB of RAM.
                    # By multiplying by 255 and casting to uint8 (integers 0-255), 
                    # we shrink the dataset to ~2.5 GB with virtually zero loss of 
                    # detail. The PyTorch DataLoader will reverse this step later.
                    tile_uint8 = (tile_norm * 255).astype(np.uint8)
                    valid_tiles.append(tile_uint8)
                    
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        
    return valid_tiles, skipped_count


def extract_tiles_from_directory(input_dir, output_file, tile_size, stride, blank_std, max_tiles):
    """Finds all images in a directory, processes them, and saves the final archive."""
    
    # Locate all .IMG or .TIF files (case-insensitive)
    img_files = glob.glob(os.path.join(input_dir, "*.[iI][mM][gG]"))
    img_files.extend(glob.glob(os.path.join(input_dir, "*.[tT][iI][fF]")))
    
    if not img_files:
        print(f"No .IMG or .TIF files found in {input_dir}")
        return

    print(f"Found {len(img_files)} images. Beginning extraction...")
    
    master_tiles = []
    total_skipped = 0
    
    # Iterate through every raster in the folder
    for img_path in img_files:
        if len(master_tiles) >= max_tiles:
            break
            
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        # Offload the heavy lifting to our processing function
        new_tiles, skipped = process_single_image(
            img_path=img_path,
            tile_size=tile_size,
            stride=stride,
            blank_std=blank_std,
            current_tile_count=len(master_tiles),
            max_tiles=max_tiles
        )
        
        master_tiles.extend(new_tiles)
        total_skipped += skipped
        
        print(f"  -> Extracted {len(new_tiles)} valid tiles (Skipped {skipped} blank)")

    # ---------------------------------------------------------
    # STEP 5: Final Packaging
    # ---------------------------------------------------------
    # Stack the list of 2D arrays into a single 3D numpy array: Shape [N, 224, 224]
    final_array = np.array(master_tiles)
    
    print("\n=== EXTRACTION COMPLETE ===")
    print(f"Valid tiles saved:   {len(final_array)}")
    print(f"Blank tiles skipped: {total_skipped}")
    
    # Save as a highly compressed .npz archive format.
    print(f"Saving dataset to {output_file}...")
    np.savez_compressed(output_file, tiles=final_array)
    
    # Print the final file footprint to confirm it will fit in Colab/Drive
    mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Done! Final archive size: {mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-image Lunar Tile Extractor")
    parser.add_argument("--dir", type=str, required=True, 
                        help="Directory containing your .IMG or .TIF rasters")
    parser.add_argument("--out", type=str, default="lunar_tiles_v2.npz", 
                        help="Output .npz filename")
    parser.add_argument("--tile-size", type=int, default=224, 
                        help="Height and width of the extracted square tiles")
    parser.add_argument("--stride", type=int, default=112, 
                        help="Step size for the extraction grid. 112 yields 50 %% overlap on a 224px tile.")
    parser.add_argument("--blank-std", type=float, default=0.02, 
                        help="Standard deviation floor. Drops tiles that are completely flat/black.")
    parser.add_argument("--max-tiles", type=int, default=50000, 
                        help="Hard limit on dataset size to prevent out-of-memory crashes.")
    
    args = parser.parse_args()
    
    extract_tiles_from_directory(
        input_dir=args.dir, 
        output_file=args.out, 
        tile_size=args.tile_size, 
        stride=args.stride, 
        blank_std=args.blank_std, 
        max_tiles=args.max_tiles
    )