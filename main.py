"""
Main script to process PIV images.

Requirements:
- Raw images in ./raw_piv
- piv_pipeline.py module with processing functions
"""

from src.piv_pipeline import batch_process_piv_images, process_with_quality_check


if __name__ == "__main__":
    INPUT_DIR = "./raw_piv"
    OUTPUT_DIR = "./masked_images"

    # # Simple batch processing (visualize only first pair)
    # print("Starting batch processing...")
    # batch_process_piv_images(
    #     input_dir=INPUT_DIR,
    #     output_dir=OUTPUT_DIR,
    #     visualize_first=True
    # )

    # Process with periodic quality checks (uncomment to use)
    print("Starting processing with quality checks...")
    process_with_quality_check(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        check_interval=2  # Show visualization every 2 images
    )
