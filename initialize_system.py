import os
import sys
import shutil
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Import the new path manager
try:
    from path_config import path_manager
except ImportError:
    # Add current directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from path_config import path_manager


def log_step(message):
    """Log initialization steps"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def create_directories():
    """Create necessary directories"""
    log_step("Creating directory structure...")

    # Directories are already created by path_manager initialization
    directories = [
        path_manager.get_data_path(),
        path_manager.get_model_path(),
        path_manager.get_logs_path(),
        path_manager.get_cache_path(),
        path_manager.get_temp_path()
    ]

    for dir_path in directories:
        if dir_path.exists():
            log_step(f"✅ Directory exists: {dir_path}")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                log_step(f"✅ Created directory: {dir_path}")
            except Exception as e:
                log_step(f"⚠️ Failed to create {dir_path}: {e}")
                return False

    # Create kaggle subdirectory
    kaggle_dir = path_manager.get_data_path('kaggle')
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"✅ Created kaggle directory: {kaggle_dir}")

    return True


def check_existing_datasets():
    """Check for existing datasets in the project structure"""
    log_step("Checking for existing datasets...")
    
    # Check for datasets in the current project structure
    base_dir = path_manager.base_paths['base']
    
    # Possible source locations
    source_locations = [
        base_dir / "data" / "kaggle" / "Fake.csv",
        base_dir / "data" / "kaggle" / "True.csv", 
        base_dir / "data" / "combined_dataset.csv"
    ]
    
    found_files = []
    for source_file in source_locations:
        if source_file.exists():
            found_files.append(source_file)
            log_step(f"✅ Found existing dataset: {source_file}")
    
    return found_files


def copy_existing_datasets():
    """Copy existing datasets if they're not in the target location"""
    log_step("Copying existing datasets to target locations...")

    base_dir = path_manager.base_paths['base']
    target_data_dir = path_manager.get_data_path()
    
    # Define source-target pairs
    copy_operations = [
        (base_dir / "data" / "kaggle" / "Fake.csv", target_data_dir / "kaggle" / "Fake.csv"),
        (base_dir / "data" / "kaggle" / "True.csv", target_data_dir / "kaggle" / "True.csv"),
        (base_dir / "data" / "combined_dataset.csv", target_data_dir / "combined_dataset.csv")
    ]
    
    copied_count = 0
    for source, target in copy_operations:
        # Skip if source and target are the same (already in correct location)
        if source == target:
            if source.exists():
                log_step(f"✅ Dataset already in correct location: {target}")
                copied_count += 1
            continue
            
        if source.exists():
            try:
                # Ensure target directory exists
                target.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source, target)
                log_step(f"✅ Copied {source} → {target}")
                copied_count += 1
            except Exception as e:
                log_step(f"⚠️ Failed to copy {source}: {e}")
        else:
            log_step(f"⚠️ Source file not found: {source}")

    return copied_count > 0


def create_minimal_dataset():
    """Create a minimal dataset if no existing dataset is found"""
    log_step("Creating minimal dataset...")

    combined_path = path_manager.get_combined_dataset_path()

    if combined_path.exists():
        log_step(f"✅ Combined dataset already exists: {combined_path}")
        return True

    try:
        # Create minimal training data with diverse examples
        minimal_data = pd.DataFrame({
            'text': [
                # Real news examples
                'Scientists at MIT have developed a new renewable energy technology that could revolutionize solar power generation.',
                'The Federal Reserve announced interest rate decisions following their latest economic review meeting.',
                'Local authorities report significant improvements in air quality following new environmental regulations.',
                'Research published in Nature journal reveals new insights about climate change adaptation strategies.',
                'Economic indicators show steady growth in the manufacturing sector across multiple regions.',
                'Healthcare officials recommend updated vaccination schedules based on latest medical research findings.',
                'Transportation department announces infrastructure improvements for major highway systems nationwide.',
                'Educational institutions implement new digital learning platforms to enhance student engagement.',
                'Agricultural experts develop drought-resistant crop varieties to improve food security globally.',
                'Technology companies invest heavily in cybersecurity measures to protect user data privacy.',
                
                # Fake news examples  
                'SHOCKING: Government officials secretly planning to control population through mind control technology.',
                'EXCLUSIVE: Celebrities caught in massive alien communication scandal that mainstream media won\'t report.',
                'BREAKING: Scientists discover time travel but government hiding the truth from public knowledge.',
                'EXPOSED: Pharmaceutical companies deliberately spreading diseases to increase their massive profits.',
                'URGENT: Social media platforms using secret algorithms to brainwash users into political compliance.',
                'LEAKED: Banking system about to collapse completely, insiders reveal financial catastrophe coming soon.',
                'CONFIRMED: Weather modification technology being used to create artificial natural disasters worldwide.',
                'REVEALED: Food companies adding dangerous chemicals that cause instant health problems and addiction.',
                'CONSPIRACY: Educational system designed to suppress critical thinking and create obedient citizens.',
                'TRUTH: Technology giants working with foreign powers to undermine national sovereignty completely.'
            ],
            'label': [
                # Real news labels (0)
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                # Fake news labels (1)  
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ]
        })

        # Save the dataset
        minimal_data.to_csv(combined_path, index=False)
        log_step(f"✅ Created minimal dataset with {len(minimal_data)} samples at {combined_path}")
        
        # Verify the file was created correctly
        if combined_path.exists():
            df_check = pd.read_csv(combined_path)
            log_step(f"✅ Verified dataset: {len(df_check)} rows loaded successfully")
            return True
        else:
            log_step("❌ Failed to verify created dataset")
            return False
            
    except Exception as e:
        log_step(f"❌ Failed to create minimal dataset: {str(e)}")
        return False


def run_initial_training():
    """Run basic model training"""
    log_step("Starting initial model training...")

    try:
        # Get all the paths
        model_path = path_manager.get_model_file_path()
        vectorizer_path = path_manager.get_vectorizer_path()
        pipeline_path = path_manager.get_pipeline_path()

        log_step(f"Model path: {model_path}")
        log_step(f"Vectorizer path: {vectorizer_path}")
        log_step(f"Pipeline path: {pipeline_path}")

        # Check if model already exists
        if pipeline_path.exists() or (model_path.exists() and vectorizer_path.exists()):
            log_step("✅ Model files already exist, checking if pipeline needs to be created...")
            
            # If individual components exist but pipeline doesn't, create pipeline
            if model_path.exists() and vectorizer_path.exists() and not pipeline_path.exists():
                log_step("Creating pipeline from existing components...")
                try:
                    import joblib
                    from sklearn.pipeline import Pipeline
                    
                    # Load existing components
                    model = joblib.load(model_path)
                    vectorizer = joblib.load(vectorizer_path)
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ('vectorizer', vectorizer),
                        ('model', model)
                    ])
                    
                    # Save pipeline
                    joblib.dump(pipeline, pipeline_path)
                    log_step(f"✅ Created pipeline from existing components: {pipeline_path}")
                    
                except Exception as e:
                    log_step(f"⚠️ Failed to create pipeline from existing components: {e}")
            
            return True

        # Import required libraries
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.pipeline import Pipeline
        import joblib

        # Load dataset
        dataset_path = path_manager.get_combined_dataset_path()
        if not dataset_path.exists():
            log_step("❌ No dataset available for training")
            return False

        df = pd.read_csv(dataset_path)
        log_step(f"Loaded dataset with {len(df)} samples")

        # Validate dataset
        if len(df) < 10:
            log_step("❌ Dataset too small for training")
            return False

        # Prepare data
        X = df['text'].values
        y = df['label'].values

        # Check class distribution
        class_counts = pd.Series(y).value_counts()
        log_step(f"Class distribution: {class_counts.to_dict()}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(class_counts) > 1 else None
        )

        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )),
            ('model', LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ))
        ])

        # Train model
        log_step("Training model...")
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Ensure model directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save complete pipeline FIRST (this is the priority)
        log_step(f"Saving pipeline to: {pipeline_path}")
        joblib.dump(pipeline, pipeline_path)
        
        # Verify pipeline was saved
        if pipeline_path.exists():
            log_step(f"✅ Pipeline saved successfully to {pipeline_path}")
            
            # Test loading the pipeline
            try:
                test_pipeline = joblib.load(pipeline_path)
                test_pred = test_pipeline.predict(["This is a test"])
                log_step(f"✅ Pipeline verification successful: {test_pred}")
            except Exception as e:
                log_step(f"⚠️ Pipeline verification failed: {e}")
        else:
            log_step(f"❌ Pipeline was not saved to {pipeline_path}")

        # Save individual components for backward compatibility
        try:
            joblib.dump(pipeline.named_steps['model'], model_path)
            joblib.dump(pipeline.named_steps['vectorizer'], vectorizer_path)
            log_step(f"✅ Saved individual components")
        except Exception as e:
            log_step(f"⚠️ Failed to save individual components: {e}")

        # Save metadata
        metadata = {
            "model_version": "v1.0_init",
            "model_type": "logistic_regression_pipeline",
            "test_accuracy": float(accuracy),
            "test_f1": float(f1),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "timestamp": datetime.now().isoformat(),
            "training_method": "initialization",
            "environment": path_manager.environment,
            "data_path": str(dataset_path),
            "class_distribution": class_counts.to_dict(),
            "pipeline_created": pipeline_path.exists(),
            "individual_components_created": model_path.exists() and vectorizer_path.exists()
        }

        metadata_path = path_manager.get_metadata_path()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        log_step(f"✅ Training completed successfully")
        log_step(f"   Accuracy: {accuracy:.4f}")
        log_step(f"   F1 Score: {f1:.4f}")
        log_step(f"   Pipeline saved: {pipeline_path.exists()}")
        log_step(f"   Model saved to: {model_path}")
        log_step(f"   Vectorizer saved to: {vectorizer_path}")
        
        return True

    except Exception as e:
        log_step(f"❌ Training failed: {str(e)}")
        import traceback
        log_step(f"❌ Traceback: {traceback.format_exc()}")
        return False


def create_initial_logs():
    """Create initial log files"""
    log_step("Creating initial log files...")

    try:
        # Activity log
        activity_log = [{
            "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
            "event": "System initialized successfully",
            "level": "INFO",
            "environment": path_manager.environment
        }]

        activity_log_path = path_manager.get_activity_log_path()
        with open(activity_log_path, 'w') as f:
            json.dump(activity_log, f, indent=2)
        log_step(f"✅ Created activity log: {activity_log_path}")

        # Create empty monitoring logs
        monitoring_log_path = path_manager.get_logs_path("monitoring_log.json")
        with open(monitoring_log_path, 'w') as f:
            json.dump([], f)
        log_step(f"✅ Created monitoring log: {monitoring_log_path}")

        # Create other necessary log files
        log_files = [
            "drift_history.json",
            "drift_alerts.json", 
            "scheduler_execution.json",
            "scheduler_errors.json"
        ]

        for log_file in log_files:
            log_path = path_manager.get_logs_path(log_file)
            if not log_path.exists():
                with open(log_path, 'w') as f:
                    json.dump([], f)
                log_step(f"✅ Created {log_file}")

        return True

    except Exception as e:
        log_step(f"❌ Log creation failed: {str(e)}")
        return False


def verify_system():
    """Verify that the system is properly initialized"""
    log_step("Verifying system initialization...")
    
    # Check critical files
    critical_files = [
        (path_manager.get_combined_dataset_path(), "Combined dataset"),
        (path_manager.get_model_file_path(), "Model file"),
        (path_manager.get_vectorizer_path(), "Vectorizer file"),
        (path_manager.get_pipeline_path(), "Pipeline file"),
        (path_manager.get_metadata_path(), "Metadata file"),
        (path_manager.get_activity_log_path(), "Activity log")
    ]
    
    all_good = True
    for file_path, description in critical_files:
        if file_path.exists():
            log_step(f"✅ {description}: {file_path}")
        else:
            log_step(f"❌ Missing {description}: {file_path}")
            if description == "Pipeline file":
                # Pipeline is critical, mark as not all good
                all_good = False
    
    # Test model loading - prioritize pipeline
    try:
        import joblib
        pipeline_path = path_manager.get_pipeline_path()
        if pipeline_path.exists():
            pipeline = joblib.load(pipeline_path)
            test_pred = pipeline.predict(["This is a test text"])
            log_step(f"✅ Pipeline test prediction successful: {test_pred}")
        else:
            log_step("⚠️ Pipeline not available, testing individual components...")
            model_path = path_manager.get_model_file_path()
            vectorizer_path = path_manager.get_vectorizer_path()
            if model_path.exists() and vectorizer_path.exists():
                model = joblib.load(model_path)
                vectorizer = joblib.load(vectorizer_path)
                test_text_vec = vectorizer.transform(["This is a test text"])
                test_pred = model.predict(test_text_vec)
                log_step(f"✅ Individual components test prediction successful: {test_pred}")
            else:
                log_step("❌ No working model components found")
                all_good = False
    except Exception as e:
        log_step(f"❌ Model test failed: {e}")
        all_good = False
    
    return all_good


def main():
    """Main initialization function"""
    log_step("🚀 Starting system initialization...")
    log_step(f"🌍 Environment: {path_manager.environment}")
    log_step(f"📁 Base directory: {path_manager.base_paths['base']}")
    log_step(f"📊 Data directory: {path_manager.base_paths['data']}")
    log_step(f"🤖 Model directory: {path_manager.base_paths['model']}")

    steps = [
        ("Directory Creation", create_directories),
        ("Existing Dataset Copy", copy_existing_datasets), 
        ("Minimal Dataset Creation", create_minimal_dataset),
        ("Model Training", run_initial_training),
        ("Log File Creation", create_initial_logs),
        ("System Verification", verify_system)
    ]

    failed_steps = []
    completed_steps = []

    for step_name, step_function in steps:
        try:
            log_step(f"🔄 Starting: {step_name}")
            if step_function():
                log_step(f"✅ {step_name} completed")
                completed_steps.append(step_name)
            else:
                log_step(f"❌ {step_name} failed")
                failed_steps.append(step_name)
        except Exception as e:
            log_step(f"❌ {step_name} failed: {str(e)}")
            failed_steps.append(step_name)

    # Summary
    log_step(f"\n📊 Initialization Summary:")
    log_step(f"   ✅ Completed: {len(completed_steps)}/{len(steps)} steps")
    log_step(f"   ❌ Failed: {len(failed_steps)}/{len(steps)} steps")
    
    if completed_steps:
        log_step(f"   Completed steps: {', '.join(completed_steps)}")
    
    if failed_steps:
        log_step(f"   Failed steps: {', '.join(failed_steps)}")
        log_step(f"⚠️ Initialization completed with {len(failed_steps)} failed steps")
    else:
        log_step("🎉 System initialization completed successfully!")

    # Environment info
    log_step(f"\n🔍 Environment Information:")
    env_info = path_manager.get_environment_info()
    log_step(f"   Environment: {env_info['environment']}")
    log_step(f"   Available datasets: {sum(env_info['available_datasets'].values())}")
    log_step(f"   Available models: {sum(env_info['available_models'].values())}")

    # Final pipeline check
    pipeline_path = path_manager.get_pipeline_path()
    log_step(f"\n🎯 Final Pipeline Check:")
    log_step(f"   Pipeline path: {pipeline_path}")
    log_step(f"   Pipeline exists: {pipeline_path.exists()}")
    if pipeline_path.exists():
        try:
            import joblib
            pipeline = joblib.load(pipeline_path)
            log_step(f"   Pipeline loadable: ✅")
            log_step(f"   Pipeline steps: {list(pipeline.named_steps.keys())}")
        except Exception as e:
            log_step(f"   Pipeline load error: {e}")

    log_step("\n🎯 System ready for use!")
    
    return len(failed_steps) == 0


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)