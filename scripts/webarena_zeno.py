
if __name__=="__main__":
    import pandas as pd
    import json
    import os
    from dotenv import load_dotenv
    import zeno_client
    import argparse
    from scripts.html2json import main

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str,defalut="")
    parser.add_argument("--config_json", type=str, default="../config_files/test.raw.json")
    args = parser.parse_args()
    main(args.result_folder, args.config_json)