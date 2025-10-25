#!/usr/bin/env python3
"""
Enhanced RecurrentGPT CLI

Command-line interface for the improved document generation system.

Usage:
    python enhanced_main.py --input "Write about drones" --output output.md
    python enhanced_main.py --input "AI ethics" --sections 8 --interactive
"""

import argparse
import os
from dotenv import load_dotenv
from enhanced_recurrentgpt import EnhancedRecurrentGPT


def main():
    parser = argparse.ArgumentParser(description="Enhanced RecurrentGPT - Intelligent Document Generation")

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Initial input/topic for document generation'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output.md',
        help='Output file path (default: output.md)'
    )

    parser.add_argument(
        '--sections', '-s',
        type=int,
        default=10,
        help='Target number of sections (default: 10)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4',
        help='Model to use for planning and critique (default: gpt-4)'
    )

    parser.add_argument(
        '--writer-model',
        type=str,
        default=None,
        help='Model to use for writing (default: same as --model)'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enable interactive mode with manual approval'
    )

    parser.add_argument(
        '--format',
        choices=['markdown', 'text', 'json', 'html'],
        default='markdown',
        help='Output format (default: markdown)'
    )

    parser.add_argument(
        '--auto-approve-threshold',
        type=int,
        default=7,
        help='Auto-approve sections with score >= threshold (default: 7)'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum regeneration attempts per section (default: 3)'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables or .env file")
        return 1

    print("="*70)
    print("Enhanced RecurrentGPT - Intelligent Document Generation")
    print("="*70)
    print(f"Topic: {args.input}")
    print(f"Target sections: {args.sections}")
    print(f"Model: {args.model}")
    print(f"Writer model: {args.writer_model or args.model}")
    print(f"Interactive mode: {args.interactive}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print("="*70 + "\n")

    # Initialize system
    print("Initializing Enhanced RecurrentGPT system...")
    engine = EnhancedRecurrentGPT(
        api_key=api_key,
        model=args.model,
        writer_model=args.writer_model
    )

    # Configure
    engine.max_regeneration_attempts = args.max_retries
    engine.auto_approve_threshold = args.auto_approve_threshold

    # Generate document
    try:
        print("\nStarting document generation...\n")

        document = engine.generate_full_document(
            initial_input=args.input,
            num_sections=args.sections,
            interactive=args.interactive
        )

        # Export
        if args.format != 'markdown':
            document = engine.export_document(format=args.format)

        # Save to file
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(document)

        print(f"\n{'='*70}")
        print(f"✓ Document generated successfully!")
        print(f"✓ Saved to: {args.output}")
        print(f"✓ Total paragraphs: {len(engine.memory.paragraphs)}")
        print(f"✓ Total claims: {sum(len(s.claims) for s in engine.memory.sections.values())}")

        # Print quality summary
        dashboard_data = engine.quality_monitor.get_dashboard_data()
        if dashboard_data.get('status') != 'no_data':
            print(f"✓ Average quality score: {dashboard_data['average_score']:.2f}/10")
            print(f"✓ Total interventions: {dashboard_data['interventions']}")
            print(f"✓ Status: {dashboard_data['status']}")

        print(f"{'='*70}\n")

        return 0

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
        print("Saving partial document...")

        document = engine.export_document(format=args.format)
        partial_output = args.output.replace('.', '_partial.')

        with open(partial_output, 'w', encoding='utf-8') as f:
            f.write(document)

        print(f"Partial document saved to: {partial_output}")
        return 1

    except Exception as e:
        print(f"\n\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
