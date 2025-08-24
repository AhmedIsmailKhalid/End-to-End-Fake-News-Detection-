#!/usr/bin/env python3
"""
Docker build validation script
Validates that the system is properly set up during container build
"""

import sys
import os

# Add app directory to Python path
sys.path.append('/app')

def main():
    """Main validation function"""
    print("🔍 Starting Docker build validation...")
    
    try:
        # Test path manager import
        from path_config import path_manager
        print(f"✅ Path manager imported successfully")
        
        # Display environment information
        print(f"🌍 Container environment: {path_manager.environment}")
        print(f"📁 Base directory: {path_manager.base_paths['base']}")
        print(f"📊 Data directory: {path_manager.base_paths['data']}")
        print(f"🤖 Model directory: {path_manager.base_paths['model']}")
        print(f"📝 Logs directory: {path_manager.base_paths['logs']}")
        
        # Check critical files
        print("\n🔍 Checking critical files...")
        critical_files = [
            (path_manager.get_combined_dataset_path(), "Combined Dataset"),
            (path_manager.get_model_file_path(), "Model File"),
            (path_manager.get_vectorizer_path(), "Vectorizer File"),
            (path_manager.get_metadata_path(), "Metadata File")
        ]
        
        files_found = 0
        for file_path, description in critical_files:
            if file_path.exists():
                print(f"✅ {description}: {file_path}")
                files_found += 1
            else:
                print(f"❌ {description}: {file_path}")
        
        # Test critical imports
        print("\n🐍 Testing Python imports...")
        imports_to_test = [
            ('pandas', 'Data processing'),
            ('sklearn', 'Machine learning'),
            ('streamlit', 'Web interface'),
            ('fastapi', 'API framework'),
            ('numpy', 'Numerical computing'),
            ('requests', 'HTTP client')
        ]
        
        import_success = 0
        for module_name, description in imports_to_test:
            try:
                __import__(module_name)
                print(f"✅ {description} ({module_name})")
                import_success += 1
            except ImportError as e:
                print(f"❌ {description} ({module_name}): {e}")
        
        # Summary
        print(f"\n📊 Validation Summary:")
        print(f"   Files found: {files_found}/{len(critical_files)}")
        print(f"   Imports successful: {import_success}/{len(imports_to_test)}")
        
        # Determine overall status
        if files_found >= 3 and import_success == len(imports_to_test):
            print("🎉 Docker build validation PASSED")
            return 0
        elif files_found >= 2 and import_success >= len(imports_to_test) - 1:
            print("⚠️ Docker build validation PASSED with warnings")
            return 0
        else:
            print("❌ Docker build validation FAILED")
            return 1
            
    except Exception as e:
        print(f"❌ Docker build validation ERROR: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())