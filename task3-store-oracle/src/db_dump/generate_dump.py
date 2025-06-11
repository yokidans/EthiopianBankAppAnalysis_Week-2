import cx_Oracle
import sqlparse
from pathlib import Path
from datetime import datetime

def get_connection():
    """Establish database connection"""
    return cx_Oracle.connect(
        user='bank_reviews',
        password='526245',
        dsn='localhost/XEPDB1'
    )

def sanitize_value(value, data_type):
    """Properly format values for SQL insertion"""
    if value is None:
        return "NULL"
    
    if isinstance(value, cx_Oracle.LOB):
        value = value.read()
    
    if 'DATE' in data_type:
        if isinstance(value, datetime):
            return "TO_DATE('" + value.strftime('%Y-%m-%d %H:%M:%S') + "', 'YYYY-MM-DD HH24:MI:SS')"
        return "TO_DATE('" + str(value) + "', 'YYYY-MM-DD HH24:MI:SS')"
    
    if 'CHAR' in data_type or 'LOB' in data_type:
        escaped = str(value).replace("'", "''").replace("\n", "\\n")
        if len(escaped) > 4000:
            escaped = escaped[:4000] + "... [TRUNCATED]"
        return "'" + escaped + "'"
    
    if 'NUMBER' in data_type:
        return str(value)
    
    return "'" + str(value).replace("'", "''") + "'"

def write_schema_dump(conn, output_dir):
    """Generate schema DDL"""
    schema_file = output_dir / "01_schema_dump.sql"
    cursor = conn.cursor()
    
    print("Generating schema dump...")
    with schema_file.open('w', encoding='utf-8') as f:
        f.write("-- Database Schema Dump\n")
        f.write("-- Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("SET DEFINE OFF;\n\n")
        
        cursor.execute("""
        SELECT DISTINCT object_type 
        FROM all_objects 
        WHERE owner = USER 
        AND object_type NOT IN ('INDEX')
        """)
        
        for obj_type in [row[0] for row in cursor]:
            f.write("\n-- " + obj_type + " definitions\n")
            cursor.execute(f"""
            SELECT DBMS_METADATA.GET_DDL('{obj_type}', object_name, owner)
            FROM all_objects
            WHERE owner = USER
            AND object_type = '{obj_type}'
            """)
            
            for row in cursor:
                ddl = row[0].read() if isinstance(row[0], cx_Oracle.LOB) else row[0]
                formatted = sqlparse.format(
                    ddl,
                    reindent=True,
                    keyword_case='upper',
                    identifier_case='lower'
                )
                f.write(formatted + ";\n\n")

def write_data_dump(conn, output_dir):
    """Generate data INSERT statements"""
    data_file = output_dir / "02_data_dump.sql"
    cursor = conn.cursor()
    
    print("Generating data dump...")
    with data_file.open('w', encoding='utf-8') as f:
        f.write("-- Database Data Dump\n")
        f.write("-- Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("SET DEFINE OFF;\n")
        f.write("BEGIN TRANSACTION;\n\n")
        
        cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
        for table in [row[0] for row in cursor]:
            f.write("\n-- Data for table " + table + "\n")
            
            cursor.execute(f"""
            SELECT column_name, data_type 
            FROM user_tab_columns 
            WHERE table_name = '{table}'
            ORDER BY column_id
            """)
            columns = cursor.fetchall()
            
            cursor.execute(f"SELECT * FROM {table}")
            for row in cursor:
                values = []
                for i, value in enumerate(row):
                    values.append(sanitize_value(value, columns[i][1]))
                
                f.write(
                    "INSERT INTO " + table + " (" + ",".join([col[0] for col in columns]) + ") " +
                    "VALUES (" + ",".join(values) + ");\n"
                )
        
        f.write("\nCOMMIT;\n")

def main():
    output_dir = Path("oracle_dump")
    output_dir.mkdir(exist_ok=True)
    
    try:
        conn = get_connection()
        write_schema_dump(conn, output_dir)
        write_data_dump(conn, output_dir)
        print("Dump completed successfully!")
    except Exception as e:
        print("Error generating dump: " + str(e))
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()