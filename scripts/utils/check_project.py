#!/usr/bin/env python3
"""
Project Structure Checker
=========================

Script kiểm tra cấu trúc project và best practices.
Chạy script này để đảm bảo project tuân thủ các quy tắc đã định.

Usage:
    python scripts/utils/check_project.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config


class ProjectChecker:
    """Kiểm tra cấu trúc project và best practices"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.issues = []
        self.warnings = []
        self.success_count = 0

    def check_directory_structure(self) -> bool:
        """Kiểm tra cấu trúc thư mục"""
        print("🔍 Checking directory structure...")

        required_dirs = [
            "app",
            "core",
            "data",
            "models",
            "indexes",
            "scripts",
            "scripts/utils",
            "logs",
            "reports",
        ]

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                print(f"  ✅ {dir_path}/")
                self.success_count += 1
            else:
                print(f"  ❌ {dir_path}/ (missing)")
                self.issues.append(f"Missing directory: {dir_path}")

        return len(self.issues) == 0

    def check_scripts_structure(self) -> bool:
        """Kiểm tra cấu trúc scripts"""
        print("\n🔍 Checking scripts structure...")

        scripts_dir = self.project_root / "scripts"

        # Check main pipeline scripts
        main_scripts = [
            "01_check_environment.py",
            "02_prepare_training_data.py",
            "03_train_models.py",
        ]

        for script in main_scripts:
            script_path = scripts_dir / script
            if script_path.exists():
                print(f"  ✅ {script}")
                self.success_count += 1
            else:
                print(f"  ❌ {script} (missing)")
                self.issues.append(f"Missing main script: {script}")

        # Check utils directory
        utils_dir = scripts_dir / "utils"
        if utils_dir.exists():
            print(f"  ✅ utils/")
            self.success_count += 1

            # Check utility files
            utility_files = ["filter_dataset.py", "run_filter.py", "__init__.py"]
            for util_file in utility_files:
                util_path = utils_dir / util_file
                if util_path.exists():
                    print(f"    ✅ {util_file}")
                    self.success_count += 1
                else:
                    print(f"    ❌ {util_file} (missing)")
                    self.warnings.append(f"Missing utility: {util_file}")
        else:
            print(f"  ❌ utils/ (missing)")
            self.issues.append("Missing utils directory")

        return len(self.issues) == 0

    def check_config_files(self) -> bool:
        """Kiểm tra config files"""
        print("\n🔍 Checking configuration files...")

        config_files = ["config.py", "run_pipeline.py", "README.md", "requirements.txt"]

        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                print(f"  ✅ {config_file}")
                self.success_count += 1
            else:
                print(f"  ❌ {config_file} (missing)")
                self.issues.append(f"Missing config file: {config_file}")

        return len(self.issues) == 0

    def check_naming_conventions(self) -> bool:
        """Kiểm tra naming conventions"""
        print("\n🔍 Checking naming conventions...")

        scripts_dir = self.project_root / "scripts"

        # Check for conflicting names
        all_files = list(scripts_dir.glob("*.py"))
        file_names = [f.stem for f in all_files]

        # Check for duplicate prefixes
        prefixes = [name[:2] for name in file_names if name[:2].isdigit()]
        duplicate_prefixes = [p for p in set(prefixes) if prefixes.count(p) > 1]

        if duplicate_prefixes:
            for prefix in duplicate_prefixes:
                print(f"  ❌ Duplicate prefix: {prefix}")
                self.issues.append(f"Duplicate prefix: {prefix}")
        else:
            print(f"  ✅ No naming conflicts")
            self.success_count += 1

        return len(self.issues) == 0

    def check_documentation(self) -> bool:
        """Kiểm tra documentation"""
        print("\n🔍 Checking documentation...")

        doc_files = [
            "README.md",
            "DEPLOYMENT_GUIDE.md",
        ]

        for doc_file in doc_files:
            doc_path = self.project_root / doc_file
            if doc_path.exists():
                print(f"  ✅ {doc_file}")
                self.success_count += 1
            else:
                print(f"  ❌ {doc_file} (missing)")
                self.warnings.append(f"Missing documentation: {doc_file}")

        return len(self.issues) == 0

    def run_all_checks(self) -> Dict[str, any]:
        """Chạy tất cả checks"""
        print("=" * 60)
        print("PROJECT STRUCTURE CHECKER")
        print("=" * 60)

        checks = [
            self.check_directory_structure,
            self.check_scripts_structure,
            self.check_config_files,
            self.check_naming_conventions,
            self.check_documentation,
        ]

        all_passed = True
        for check in checks:
            if not check():
                all_passed = False

        # Summary
        print("\n" + "=" * 60)
        print("CHECK SUMMARY")
        print("=" * 60)
        print(f"✅ Success checks: {self.success_count}")
        print(f"❌ Issues found: {len(self.issues)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")

        if self.issues:
            print("\n❌ ISSUES TO FIX:")
            for issue in self.issues:
                print(f"  - {issue}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if all_passed and not self.warnings:
            print("\n🎉 All checks passed! Project structure is optimal.")
        elif all_passed:
            print("\n✅ All critical checks passed. Some warnings to consider.")
        else:
            print("\n❌ Some issues need to be fixed before proceeding.")

        return {
            "passed": all_passed,
            "success_count": self.success_count,
            "issues": self.issues,
            "warnings": self.warnings,
        }


def main():
    """Main function"""
    checker = ProjectChecker()
    result = checker.run_all_checks()

    if result["passed"]:
        print("\n✅ Project structure check completed successfully!")
        return True
    else:
        print("\n❌ Project structure check failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
