from vlib.infrastructure.endpoint import app

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='127.0.0.1', port=5001)