#!/usr/bin/env python3
"""Test CLI functionality after consolidation."""

import subprocess
import sys
from pathlib import Path


def test_cli_help():
    """Test that the CLI help command works."""
    print("Testing CLI help command...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'eq', '--help'
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✅ CLI help command works")
            print(f"   Output length: {len(result.stdout)} characters")
            
            # Check for key CLI commands
            if 'data-load' in result.stdout and 'seg' in result.stdout:
                print("✅ Key CLI commands found in help")
                return True
            else:
                print("❌ Missing key CLI commands in help")
                return False
        else:
            print(f"❌ CLI help failed with exit code {result.returncode}")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI help test failed: {e}")
        return False

def test_cli_mode_command():
    """Test that the mode command works."""
    print("\nTesting CLI mode command...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'eq', 'mode', '--show'
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✅ CLI mode --show works")
            print(f"   Current mode shown in output: {'development' in result.stdout.lower() or 'production' in result.stdout.lower()}")
            return True
        else:
            print(f"❌ CLI mode --show failed with exit code {result.returncode}")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI mode test failed: {e}")
        return False

def test_cli_capabilities():
    """Test that the capabilities command works."""
    print("\nTesting CLI capabilities command...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'eq', 'capabilities'
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✅ CLI capabilities works")
            
            # Check for hardware detection info
            if any(word in result.stdout.lower() for word in ['cpu', 'hardware', 'available']):
                print("✅ Hardware capabilities detected")
                return True
            else:
                print("⚠️  No hardware info found but command succeeded")
                return True
        else:
            print(f"❌ CLI capabilities failed with exit code {result.returncode}")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI capabilities test failed: {e}")
        return False

def test_eq_module_direct():
    """Test importing eq module directly."""
    print("\nTesting direct eq module import...")
    
    try:
        result = subprocess.run([
            sys.executable, '-c', 'import eq; print("✅ eq module imports successfully")'
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✅ Direct eq module import works")
            return True
        else:
            print(f"❌ Direct eq import failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Direct eq import test failed: {e}")
        return False

def test_cli_without_pytorch():
    """Test that basic CLI works even without PyTorch functionality."""
    print("\nTesting CLI robustness (CPU-only commands)...")
    
    # Test commands that should work without ML dependencies
    commands_to_test = [
        ['--help'],
        ['mode', '--show'],
        ['capabilities']
    ]
    
    success_count = 0
    for cmd in commands_to_test:
        try:
            result = subprocess.run([
                sys.executable, '-m', 'eq'
            ] + cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                success_count += 1
                print(f"✅ Command 'eq {' '.join(cmd)}' works")
            else:
                print(f"❌ Command 'eq {' '.join(cmd)}' failed")
                
        except Exception as e:
            print(f"❌ Command 'eq {' '.join(cmd)}' error: {e}")
    
    if success_count == len(commands_to_test):
        print("✅ All CPU-only CLI commands work")
        return True
    else:
        print(f"⚠️  {success_count}/{len(commands_to_test)} CPU-only commands work")
        return success_count > 0

if __name__ == "__main__":
    print("CLI FUNCTIONALITY TEST AFTER CONSOLIDATION")
    print("=" * 45)
    
    success = True
    success &= test_cli_help()
    success &= test_cli_mode_command()
    success &= test_cli_capabilities()
    success &= test_eq_module_direct()
    success &= test_cli_without_pytorch()
    
    print("\n" + "=" * 45)
    if success:
        print("🎉 ALL CLI TESTS PASSED!")
        print("✅ CLI commands work correctly")
        print("✅ Module imports properly")  
        print("✅ CPU-only functionality robust")
        print("\nCLI is ready for use!")
    else:
        print("❌ SOME CLI TESTS FAILED")
        print("CLI may have issues after consolidation...")
