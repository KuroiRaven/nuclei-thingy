Spot detection pseudo-code:
  1. Take all nuclei regions (mask-bulle ?)
  2. Find the elbow value of brightness
  3. Define an elbow value for each nucleus region on each slice
  4. Detect spots in these regions
  5. Cleanup lone elements

Min-dist-all:
  1. Nearest-neighbors first (50 ?)
  2. Test min dist avec les 50 nearest
  3. ignorer ceux dont les 50 plus proches sont plus loin que X