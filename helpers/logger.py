import logging
import sys
import json
from datetime import datetime

class ReadableFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{record.levelname}] {record.getMessage()}"
        
        if hasattr(record, 'extra_data'):
            extra = record.extra_data
            if isinstance(extra, dict):
                has_complex = any(isinstance(v, (dict, list)) for v in extra.values())
                
                if has_complex:
                    log_msg += ":"
                    for k, v in extra.items():
                        if isinstance(v, (dict, list)):
                            try:
                                val_str = json.dumps(v, indent=2)
                                val_str = "\n    ".join(val_str.splitlines())
                                log_msg += f"\n  {k}: {val_str}"
                            except (TypeError, ValueError):
                                log_msg += f"\n  {k}: {v}"
                        else:
                            log_msg += f"\n  {k}: {v}"
                else:
                    items = [f"{k}: {v}" for k, v in extra.items()]
                    log_msg += " | " + " | ".join(items)
            else:
                log_msg += f" | {extra}"
            
        return log_msg

def setup_logger(name="dl_project", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ReadableFormatter())
        logger.addHandler(handler)
        
    return logger
