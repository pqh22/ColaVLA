#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple script to check server status."""

import requests
import time
import sys

def check_server(host="localhost", port=39919):
    """Check if the server is running and responding."""
    base_url = f"http://{host}:{port}"
    
    print(f"�� Checking server at {base_url}")
    
    # Test root endpoint
    try:
        print("Testing / endpoint...")
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Root endpoint working!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test alive endpoint
    try:
        print("\nTesting /alive endpoint...")
        response = requests.get(f"{base_url}/alive", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Alive endpoint working!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Alive endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Alive endpoint error: {e}")
    
    # Test docs endpoint
    try:
        print("\nTesting /docs endpoint...")
        response = requests.get(f"{base_url}/docs", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Docs endpoint working!")
        else:
            print(f"❌ Docs endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Docs endpoint error: {e}")

if __name__ == "__main__":
    port = 39919  # Use the port from your logs
    check_server(port=port)