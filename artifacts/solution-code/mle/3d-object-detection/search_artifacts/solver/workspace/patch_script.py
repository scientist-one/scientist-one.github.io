import re

with open('solution_02c95fb7/main.py', 'r') as f:
    content = f.read()

# Replace map_path with map_img in __init__
content = content.replace("def __init__(self, lyft_data, sample_tokens, map_path, is_train=True):",
                          "def __init__(self, lyft_data, sample_tokens, map_img, is_train=True):")

content = content.replace("""        Image.MAX_IMAGE_PIXELS = None
        self.map_img = np.array(Image.open(map_path))""",
                          "        self.map_img = map_img")

# Inside main() load map_img once
main_repl = """    print("Loading Dataset...")
    lyft = LyftDataset(data_path='/workspace/lyft_data', json_path='/workspace/lyft_data/train_data', verbose=False)
    Image.MAX_IMAGE_PIXELS = None
    shared_map_img = np.array(Image.open('/workspace/lyft_data/maps/map_raster_palo_alto.png'))
    train_df = pd.read_csv('/workspace/metadata/train_ids.csv')"""

content = re.sub(r'    print\("Loading Dataset\.\.\."\)\n    lyft = .*?\n    train_df = .*?', main_repl, content, flags=re.DOTALL)

content = content.replace("'/workspace/lyft_data/maps/map_raster_palo_alto.png'", "shared_map_img")
content = content.replace("'/workspace/lyft_data_test/maps/map_raster_palo_alto.png'", "shared_map_img")

with open('solution_02c95fb7/main.py', 'w') as f:
    f.write(content)
