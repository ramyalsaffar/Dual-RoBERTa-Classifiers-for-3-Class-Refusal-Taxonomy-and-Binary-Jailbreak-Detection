# Analyze Pipeline
#-----------------
# Standalone analysis script for 3-Class Refusal Classifier.
# Loads trained models and runs comprehensive analysis on test data.
#
# Usage:
#   python src/Analyze.py                                    # Interactive mode
#   python src/Analyze.py --help                             # Show help
#   python src/Analyze.py --refusal-model models/refusal.pt  # Analyze with custom refusal model
#   python src/Analyze.py --refusal-model models/refusal.pt --jailbreak-model models/jailbreak.pt
#   python src/Analyze.py --test-data data/splits/test.pkl   # Use custom test data
#
###############################################################################


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Analyze trained Refusal Classifier and Jailbreak Detector models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode (default)
  python src/Analyze.py

  # Analyze with default model paths
  python src/Analyze.py --auto

  # Specify refusal model only (jailbreak uses default)
  python src/Analyze.py --refusal-model models/my_refusal.pt

  # Specify both models
  python src/Analyze.py --refusal-model models/refusal.pt --jailbreak-model models/jailbreak.pt

  # Use custom test data
  python src/Analyze.py --test-data data/custom_test.pkl

  # Generate PDF reports
  python src/Analyze.py --auto --generate-report
  python src/Analyze.py --auto --generate-report --report-type performance
  python src/Analyze.py --auto --generate-report --report-type interpretability
  python src/Analyze.py --auto --generate-report --report-type executive

  # Combine options
  python src/Analyze.py --refusal-model models/refusal.pt --test-data data/test.pkl --generate-report
        '''
    )

    parser.add_argument(
        '--refusal-model',
        type=str,
        default=None,
        help='Path to trained refusal classifier model (.pt file)'
    )

    parser.add_argument(
        '--jailbreak-model',
        type=str,
        default=None,
        help='Path to trained jailbreak detector model (.pt file)'
    )

    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test data (.pkl file). If not specified, uses data/splits/test.pkl'
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help='Automatically use default model paths without prompting'
    )

    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate comprehensive PDF report after analysis'
    )

    parser.add_argument(
        '--report-type',
        type=str,
        choices=['performance', 'interpretability', 'executive', 'all'],
        default='all',
        help='Type of report to generate (default: all)'
    )

    return parser.parse_args()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_file_exists(file_path: str, file_type: str) -> bool:
    """
    Validate that a file exists.

    Args:
        file_path: Path to file
        file_type: Description of file type (for error messages)

    Returns:
        True if valid, False otherwise
    """
    if file_path and not os.path.exists(file_path):
        print(f"❌ Error: {file_type} not found at {file_path}")
        return False
    return True


# =============================================================================
# REPORT GENERATION
# =============================================================================

def _generate_reports(analysis_results: Dict, report_type: str):
    """
    Generate PDF reports using ReportGenerator.

    Args:
        analysis_results: Analysis results from ExperimentRunner
        report_type: Type of report ('performance', 'interpretability', 'executive', 'all')
    """
    timestamp = get_timestamp('file')

    report_generator = ReportGenerator(class_names=CLASS_NAMES)

    # Load visualization figures

    if report_type in ['performance', 'all']:
        print("\n📊 Generating Performance Report...")

        # Load figures
        cm_fig = plt.figure(figsize=(10, 8))
        cm_img = mpimg.imread(os.path.join(reporting_viz_path, "confusion_matrix.png"))
        plt.imshow(cm_img)
        plt.axis('off')

        training_curves_fig = plt.figure(figsize=(12, 5))
        # Note: Training curves would need to be loaded from history
        plt.text(0.5, 0.5, "Training curves not available in analysis-only mode",
                ha='center', va='center', fontsize=14)
        plt.axis('off')

        class_dist_fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Class distribution: See per_class_f1.png",
                ha='center', va='center', fontsize=14)
        plt.axis('off')


        # Build metrics for the refusal classifier report from analysis_results
        metrics = build_refusal_metrics_from_results(analysis_results)


        # Add per-class metrics
        for i, class_name in enumerate(analysis_results['metadata']['refusal_class_names']):
            class_metrics = analysis_results['per_model'].get(class_name, {})
            metrics[f'class_{i}_precision'] = class_metrics.get('precision', 0)
            metrics[f'class_{i}_recall'] = class_metrics.get('recall', 0)
            metrics[f'class_{i}_f1'] = class_metrics.get('f1-score', 0)
            metrics[f'class_{i}_support'] = class_metrics.get('support', 0)


        output_path = os.path.join(reports_path, f"performance_report_{timestamp}.pdf")
        report_generator.generate_model_performance_report(
            model_name="Refusal Classifier",
            metrics=metrics,
            confusion_matrix_fig=cm_fig,
            training_curves_fig=training_curves_fig,
            class_distribution_fig=class_dist_fig,
            output_path=output_path
        )
        plt.close('all')

    # DEBUG - ADD THIS BEFORE LINE 183:
    print(f"\n🔍 DEBUG: analysis_results keys: {analysis_results.keys()}")
    print(f"🔍 DEBUG: 'jailbreak_results' in analysis_results: {'jailbreak_results' in analysis_results}")

    # JAILBREAK REPORT
    if 'jailbreak_results' in analysis_results:
        print("\n📊 Generating Jailbreak Classifier Report...")
        
        # Use SAME report_generator, just different class names
        jailbreak_report_gen = ReportGenerator(
            class_names=['Jailbreak Failed', 'Jailbreak Succeeded']
        )
        
        jailbreak_metrics = build_jailbreak_metrics_from_results(
            analysis_results['jailbreak_results']
        )
        
        # Same figure pattern as refusal
        jb_cm_fig = plt.figure(figsize=(10, 8))
        if os.path.exists(os.path.join(reporting_viz_path, "jailbreak_confusion_matrix.png")):
            jb_cm_img = mpimg.imread(os.path.join(reporting_viz_path, "jailbreak_confusion_matrix.png"))
            plt.imshow(jb_cm_img)
        else:
            plt.text(0.5, 0.5, "Jailbreak confusion matrix not available", ha='center', va='center')
        plt.axis('off')
        
        jb_training_fig = plt.figure(figsize=(12, 5))
        plt.text(0.5, 0.5, "Training curves not available", ha='center', va='center', fontsize=14)
        plt.axis('off')
        
        jb_dist_fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Class distribution: 2 classes", ha='center', va='center', fontsize=14)
        plt.axis('off')
        
        jb_output_path = os.path.join(reports_path, f"jailbreak_performance_report_{timestamp}.pdf")
        jailbreak_report_gen.generate_model_performance_report(
            model_name="Jailbreak Classifier",
            metrics=jailbreak_metrics,
            confusion_matrix_fig=jb_cm_fig,
            training_curves_fig=jb_training_fig,
            class_distribution_fig=jb_dist_fig,
            output_path=jb_output_path
        )
        plt.close('all')
        print(f"✅ Jailbreak report: {jb_output_path}")
    
    

    if report_type in ['executive', 'all']:
        print("\n📊 Generating Executive Summary...")

        # Key metrics
        key_metrics = {
            'Overall Accuracy': f"{analysis_results['confidence'].get('overall_accuracy', 0):.2%}",
            'Weighted F1 Score': f"{analysis_results['per_model'].get('analysis', {}).get('weighted_avg', {}).get('f1-score', 0):.4f}",
            'Test Samples': analysis_results['metadata']['num_test_samples'],
            'Avg Confidence': f"{analysis_results['confidence'].get('avg_confidence', 0):.4f}",
        }

        # Performance chart
        perf_fig = plt.figure(figsize=(10, 6))
        perf_img = mpimg.imread(os.path.join(reporting_viz_path, "per_class_f1.png"))
        plt.imshow(perf_img)
        plt.axis('off')

        # Recommendations
        recommendations = [
            "Model shows strong performance on test set",
            "Monitor confidence distributions in production",
            "Consider retraining if accuracy drops below 85%",
            "Implement A/B testing before major model updates"
        ]

        output_path = os.path.join(reports_path, f"executive_summary_{timestamp}.pdf")
        report_generator.generate_executive_summary(
            model_name="Refusal Classifier",
            key_metrics=key_metrics,
            performance_chart_fig=perf_fig,
            recommendations=recommendations,
            output_path=output_path
        )
        plt.close('all')

    print(f"\n✅ Reports saved to: {reports_path}")


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode():
    """Run analysis in interactive mode with user prompts."""
    print()
    print_banner("📊 REFUSAL CLASSIFIER - INTERACTIVE ANALYSIS", width=60)
    print()

    print("This will analyze BOTH classifiers:")
    print("  1. Refusal Classifier (3 classes)")
    print("  2. Jailbreak Detector (2 classes)")

    print()
    print_banner("Model Configuration", width=60, char="-")

    # Get refusal model path
    print("\nRefusal Classifier Model:")
    print("  Press Enter to use default path")
    print(f"  Default: models/{EXPERIMENT_CONFIG['experiment_name']}_refusal_best.pt")
    refusal_path = input("  Custom path: ").strip()
    if not refusal_path:
        refusal_path = None

    # Validate refusal model
    if refusal_path and not validate_file_exists(refusal_path, "Refusal model"):
        return

    # Get jailbreak model path
    print("\nJailbreak Detector Model:")
    print("  Press Enter to use default path")
    print(f"  Default: models/{EXPERIMENT_CONFIG['experiment_name']}_jailbreak_best.pt")
    jailbreak_path = input("  Custom path: ").strip()
    if not jailbreak_path:
        jailbreak_path = None

    # Validate jailbreak model
    if jailbreak_path and not validate_file_exists(jailbreak_path, "Jailbreak model"):
        return

    # Get test data path
    print("\nTest Data:")
    print("  Press Enter to use default path")
    print(f"  Default: {os.path.join(data_splits_path, 'test.pkl')}")
    test_data_path = input("  Custom path: ").strip()
    if not test_data_path:
        test_data_path = None

    # Validate test data
    if test_data_path and not validate_file_exists(test_data_path, "Test data"):
        return

    # Ask about report generation
    print("\nReport Generation:")
    generate_report = input("  Generate PDF report? (y/n, default: n): ").strip().lower() == 'y'

    report_type = 'all'
    if generate_report:
        print("  Report types:")
        print("    1. Performance only")
        print("    2. Interpretability only")
        print("    3. Executive summary only")
        print("    4. All reports (default)")
        choice = input("  Select (1-4, default: 4): ").strip()
        report_type_map = {'1': 'performance', '2': 'interpretability', '3': 'executive', '4': 'all'}
        report_type = report_type_map.get(choice, 'all')

    # Confirm and run
    print("\n" + "-"*60)
    print("Ready to analyze:")
    print(f"  Refusal model: {refusal_path or 'default'}")
    print(f"  Jailbreak model: {jailbreak_path or 'default'}")
    print(f"  Test data: {test_data_path or 'default'}")
    print(f"  Generate report: {'Yes (' + report_type + ')' if generate_report else 'No'}")
    print("-"*60)

    confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Analysis cancelled.")
        return

    # Run analysis
    runner = ExperimentRunner()

    # Run analysis with custom test data if provided
    # WHY: Allows validation on different test sets without retraining
    if test_data_path:
        print(f"\n✓ Using custom test data: {test_data_path}")

    try:
        analysis_results = runner.analyze_only(refusal_path, jailbreak_path, test_data_path)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("   Please check model paths and test data")
        print("\n🔍 FULL TRACEBACK:")
        traceback.print_exc()
        return

    # Generate report if requested
    if generate_report and analysis_results:
        print()
        print_banner("📄 GENERATING PDF REPORT", width=60)
        print(f"\nReport type: {report_type}")
        print(f"Output location: {reports_path}")

        try:
            _generate_reports(analysis_results, report_type)
        except Exception as e:
            print(f"\n❌ Report generation failed: {e}")
            print("   Analysis results are still available in results/ directory")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Parse arguments (empty when called via exec())
    args = parse_arguments()

    # If --auto flag or any arguments provided, use command-line mode
    # OR if no arguments at all (exec() case), default to auto mode
    if args.auto or args.refusal_model or args.jailbreak_model or args.test_data or len(sys.argv) == 1:
        print()
        print_banner("📊 REFUSAL CLASSIFIER - ANALYSIS MODE", width=60)
        print()

        # Use predefined paths if not specified
        # IMPORTANT: Find LATEST trained model, not current timestamp!
        if not args.refusal_model:
            # Find latest refusal model
            refusal_pattern = os.path.join(models_path, "*_refusal_best.pt")
            refusal_models = sorted(glob.glob(refusal_pattern), key=os.path.getmtime, reverse=True)
            if refusal_models:
                args.refusal_model = refusal_models[0]  # Most recent
            else:
                args.refusal_model = os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_refusal_best.pt")
        
        if not args.jailbreak_model:
            # Find latest jailbreak model
            jailbreak_pattern = os.path.join(models_path, "*_jailbreak_best.pt")
            jailbreak_models = sorted(glob.glob(jailbreak_pattern), key=os.path.getmtime, reverse=True)
            if jailbreak_models:
                args.jailbreak_model = jailbreak_models[0]  # Most recent
            else:
                args.jailbreak_model = os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_jailbreak_best.pt")
        
        if not args.test_data:
            # Find latest test split
            test_pattern = os.path.join(data_splits_path, 'test*.pkl')
            test_files = sorted(glob.glob(test_pattern), key=os.path.getmtime, reverse=True)
            if test_files:
                args.test_data = test_files[0]  # Most recent
            else:
                args.test_data = os.path.join(data_splits_path, 'test.pkl')

        # Validate file paths
        if not validate_file_exists(args.refusal_model, "Refusal model"):
            sys.exit(1)
        if not validate_file_exists(args.jailbreak_model, "Jailbreak model"):
            sys.exit(1)
        if not validate_file_exists(args.test_data, "Test data"):
            sys.exit(1)

        # Show configuration
        print("\nConfiguration:")
        print(f"  Refusal model: {args.refusal_model}")
        print(f"  Jailbreak model: {args.jailbreak_model}")
        print(f"  Test data: {args.test_data}")
        print(f"  Generate report: {'Yes (' + args.report_type + ')' if args.generate_report else 'No'}")

        # Notify if custom test data provided
        if args.test_data != os.path.join(data_splits_path, 'test.pkl'):
            print(f"\n✓ Using custom test data: {args.test_data}")

        # Run analysis
        runner = ExperimentRunner()
        
        try:
            analysis_results = runner.analyze_only(args.refusal_model, args.jailbreak_model, args.test_data)
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            print("   Please check model paths and test data")
            sys.exit(1)

        # Generate report if requested
        if args.generate_report and analysis_results:
            print()
            print_banner("📄 GENERATING PDF REPORT", width=60)
            print(f"\nReport type: {args.report_type}")
            print(f"Output location: {reports_path}")

            try:
                _generate_reports(analysis_results, args.report_type)
            except Exception as e:
                print(f"\n❌ Report generation failed: {e}")
                print("   Analysis results are still available in results/ directory")

    else:
        # Interactive mode (only if explicitly running with -i or --interactive flag)
        interactive_mode()


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
