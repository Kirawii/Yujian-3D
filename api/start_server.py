#!/usr/bin/env python3
"""
启动 SOKE Sign Language API 服务器
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
        workers=1,
        log_level="info"
    )
