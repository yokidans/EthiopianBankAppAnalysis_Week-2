# Oracle XE Setup Guide

## Windows Installation
1. Download Oracle XE from [Oracle website](https://www.oracle.com/database/technologies/xe-downloads.html)
2. Run installer with administrative privileges
3. Follow wizard instructions
4. Set SYSTEM password when prompted
5. Verify installation:
   ```bash
   sqlplus SYSTEM/yourpassword@localhost/XEPDB1