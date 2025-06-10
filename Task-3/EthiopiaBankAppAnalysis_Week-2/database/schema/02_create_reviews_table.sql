-- Create reviews table with foreign key to banks
CREATE TABLE reviews (
    review_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bank_id NUMBER NOT NULL,
    review_date DATE NOT NULL,
    reviewer_name VARCHAR2(100),
    review_title VARCHAR2(255),
    review_text CLOB,
    rating NUMBER(2,1) CHECK (rating BETWEEN 1 AND 5),
    sentiment_score NUMBER(5,3),
    sentiment_magnitude NUMBER(5,3),
    cleaned_text CLOB,
    topics VARCHAR2(255),
    CONSTRAINT fk_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
);

-- Create indexes for performance
CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX idx_reviews_date ON reviews(review_date);
CREATE INDEX idx_reviews_rating ON reviews(rating);

-- Comments for documentation
COMMENT ON TABLE reviews IS 'Contains customer reviews for banks';
COMMENT ON COLUMN reviews.sentiment_score IS 'Sentiment analysis score (-1 to 1)';
COMMENT ON COLUMN reviews.cleaned_text IS 'Preprocessed review text for analysis';