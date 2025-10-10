import json
from datetime import datetime

def handler(request):
    """Health check endpoint for ML service"""
    try:
        health_data = {
            'status': 'healthy',
            'service': 'ml-prediction',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(health_data)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }

# Vercel serverless function entry point
def lambda_handler(event, context):
    return handler(event)