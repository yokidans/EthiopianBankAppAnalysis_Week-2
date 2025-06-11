-- Create banks table
CREATE TABLE banks (
    bank_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bank_name VARCHAR2(100) NOT NULL,
    bank_code VARCHAR2(20),
    website_url VARCHAR2(255),
    established_date DATE,
    headquarters VARCHAR2(100),
    last_updated TIMESTAMP DEFAULT SYSTIMESTAMP
);

-- Comments for documentation
COMMENT ON TABLE banks IS 'Contains information about financial institutions';
COMMENT ON COLUMN banks.bank_name IS 'Official name of the bank';
COMMENT ON COLUMN banks.bank_code IS 'Unique identifier code for the bank';