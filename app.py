import os
import psycopg2
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Using the same environment variables as populate_db.py
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "spotify_recommender")
DB_USER = os.environ.get("DB_USER", "khai")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

def get_db_connection():
    """Establishes a connection to the database."""
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn


@app.route('/recommend/<string:spotify_track_id>', methods=['GET'])
def recommend(spotify_track_id):
    """
    The main recommendation endpoint.
    Takes a Spotify track ID and returns a list of the top 20 most similar songs.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # --- Step 1: Get the embedding vector for the query track ---
        cur.execute(
            """
            SELECT e.embedding_vector
            FROM embeddings e
            JOIN tracks t ON e.track_id = t.id
            WHERE t.spotify_track_id = %s
            AND e.embedding_type = 'librosa_v1'
            LIMIT 1;
            """,
            (spotify_track_id,)
        )
        
        result = cur.fetchone()
        if result is None:
            return jsonify({"error": f"Track with Spotify ID '{spotify_track_id}' not found in our library."}), 404
        
        query_vector = result[0]

        # --- Step 2: Perform the similarity search using pgvector ---
        # The <-> operator is the Euclidean distance operator from pgvector
        cur.execute(
            """
            SELECT t.spotify_track_id, t.track_name, t.artist_name, t.spotify_url, e.embedding_vector <-> %s AS distance
            FROM embeddings e
            JOIN tracks t ON e.track_id = t.id
            WHERE e.embedding_type = 'librosa_v1'
            AND t.spotify_track_id != %s
            ORDER BY distance ASC
            LIMIT 20;
            """,
            (query_vector, spotify_track_id)
        )
        
        recommendations = cur.fetchall()

        # --- Step 3: Format the results into a JSON response ---
        recommendation_list = []
        for row in recommendations:
            recommendation_list.append({
                "spotify_track_id": row[0],
                "track_name": row[1],
                "artist_name": row[2],
                "spotify_url": row[3],
                "distance": row[4]
            })

        cur.close()
        return jsonify(recommendation_list)

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return jsonify({"error": "A database error occurred."}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500
    finally:
        if conn:
            conn.close()

# --- Main execution ---
if __name__ == '__main__':
    # To run this app:
    # 1. Set the same DB environment variables as for populate_db.py
    # 2. Run the command: flask --app app run
    # 3. Open your browser to http://127.0.0.1:5000/recommend/some_spotify_id
    app.run(debug=True)
