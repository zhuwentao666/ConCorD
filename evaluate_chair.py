# evaluate_chair.py
import argparse
import os
import pickle
import sys
import nltk

# --- NLTK Data Check ---
# Ensure necessary NLTK data is available
nltk.data.find('tokenizers/punkt')
nltk.data.find('taggers/averaged_perceptron_tagger')
nltk.data.find('corpora/wordnet')

# --- Import from chair.py ---
# Assumes chair.py is in the same directory or Python path
try:
    from chair import CHAIR, print_metrics, save_hallucinated_words
except ImportError:
    print("Error: Could not import 'CHAIR' class from chair.py.")
    print("Please ensure chair.py is in the current directory or accessible via PYTHONPATH.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate caption hallucination using the CHAIR metric.")

    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input JSONL file. Each line should be a JSON object "
                             "with 'id' (image_id) and 'response' (caption) keys.")
    parser.add_argument("--coco_path", type=str, default=None,
                        help="Path to COCO annotations directory (e.g., '.../annotations_trainval2014/annotations'). "
                             "Required only if the cache file doesn't exist.")
    parser.add_argument("--cache_path", type=str, default="chair_cache.pkl",
                        help="Path to save/load the pre-initialized CHAIR evaluator object cache.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional path to save the detailed CHAIR evaluation results (per sentence) to a JSON file.")

    args = parser.parse_args()

    # --- Load or Build CHAIR Evaluator ---
    evaluator = None
    if os.path.exists(args.cache_path):
        try:
            with open(args.cache_path, 'rb') as f:
                evaluator = pickle.load(f)
            print(f"Loaded CHAIR evaluator from cache: {args.cache_path}")
        except Exception as e:
            print(f"Warning: Failed to load cache file '{args.cache_path}'. Will rebuild. Error: {e}")
            evaluator = None # Ensure evaluator is None if loading failed

    if evaluator is None:
        print(f"Cache not found or failed to load. Building CHAIR evaluator from scratch...")
        if not args.coco_path or not os.path.isdir(args.coco_path):
            parser.error(f"--coco_path ('{args.coco_path}') is required and must be a valid directory when cache ('{args.cache_path}') does not exist.")
        try:
            evaluator = CHAIR(args.coco_path)
            print(f"Successfully built evaluator.")
            try:
                with open(args.cache_path, 'wb') as f:
                    pickle.dump(evaluator, f)
                print(f"Cached evaluator to: {args.cache_path}")
            except Exception as e:
                 print(f"Warning: Failed to save cache to '{args.cache_path}'. Error: {e}")
        except Exception as e:
            print(f"Error: Failed to initialize CHAIR evaluator. Check coco_path and annotations. Error: {e}")
            sys.exit(1)

    # --- Run Evaluation ---
    print(f"\nStarting CHAIR evaluation for: {args.input_file}")
    # Use the specific keys from the user's JSONL format
    image_id_key = "id"
    caption_key = "response"

    try:
        results = evaluator.compute_chair(args.input_file, image_id_key, caption_key)
    except FileNotFoundError:
         print(f"Error: Input file not found: {args.input_file}")
         sys.exit(1)
    except Exception as e:
         print(f"Error during CHAIR computation: {e}")
         # Potentially add more specific error handling if needed based on chair.py's behavior
         sys.exit(1)

    # --- Print Overall Metrics ---
    print("\n--- Overall CHAIR Metrics ---")
    if 'overall_metrics' in results:
        print_metrics(results) # Use the print function from chair.py
    else:
        print("Warning: 'overall_metrics' not found in results.")

    # --- Save Detailed Results (Optional) ---
    if args.output_file:
        print(f"\nSaving detailed results to: {args.output_file}")
        try:
            # Use the save function from chair.py
            save_hallucinated_words(args.output_file, results)
            print("Results saved successfully.")
        except Exception as e:
            print(f"Error: Failed to save results to {args.output_file}. Error: {e}")

if __name__ == '__main__':
    main()