-- Database Schema Dump
-- Generated: 2025-06-11 23:45:16

SET DEFINE OFF;


-- TABLE definitions

CREATE TABLE "BANK_REVIEWS"."BANKS"
  ("BANK_ID" NUMBER GENERATED always AS IDENTITY MINVALUE 1 MAXVALUE 9999999999999999999999999999 INCREMENT BY 1 START WITH 1 CACHE 20
   NOORDER NOCYCLE nokeep noscale NOT NULL ENABLE,
                                           "BANK_NAME" varchar2(100) NOT NULL ENABLE,
                                                                              "BANK_CODE" varchar2(20),
                                                                                          "WEBSITE_URL" varchar2(255),
                                                                                                        "ESTABLISHED_DATE" DATE, "HEADQUARTERS" varchar2(100),
                                                                                                                                                "LAST_UPDATED" TIMESTAMP (6) DEFAULT systimestamp,
                                                                                                                                                                                     PRIMARY KEY ("BANK_ID") USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 compute STATISTICS storage(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645 PCTINCREASE 0 FREELISTS 1 FREELIST groups 1 buffer_pool DEFAULT flash_cache DEFAULT cell_flash_cache DEFAULT) TABLESPACE "USERS" ENABLE) SEGMENT creation IMMEDIATE PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS logging storage(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645 PCTINCREASE 0 FREELISTS 1 FREELIST groups 1 buffer_pool DEFAULT flash_cache DEFAULT cell_flash_cache DEFAULT) TABLESPACE "USERS";


CREATE TABLE "BANK_REVIEWS"."REVIEWS"
  ("REVIEW_ID" NUMBER GENERATED always AS IDENTITY MINVALUE 1 MAXVALUE 9999999999999999999999999999 INCREMENT BY 1 START WITH 1 CACHE 20
   NOORDER NOCYCLE nokeep noscale NOT NULL ENABLE,
                                           "BANK_ID" NUMBER NOT NULL ENABLE,
                                                                     "REVIEW_DATE" DATE NOT NULL ENABLE,
                                                                                                 "REVIEWER_NAME" varchar2(100),
                                                                                                                 "REVIEW_TITLE" varchar2(255),
                                                                                                                                "REVIEW_TEXT" CLOB,
                                                                                                                                              "RATING" number(2, 1),
                                                                                                                                                       "SENTIMENT_SCORE" number(5, 3),
                                                                                                                                                                         "SENTIMENT_MAGNITUDE" number(5, 3),
                                                                                                                                                                                               "CLEANED_TEXT" CLOB,
                                                                                                                                                                                                              "TOPICS" varchar2(255),
                                                                                                                                                                                                                       CHECK (rating BETWEEN 1 AND 5) ENABLE,
                                                                                                                                                                                                                                                      PRIMARY KEY ("REVIEW_ID") USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 compute STATISTICS storage(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645 PCTINCREASE 0 FREELISTS 1 FREELIST groups 1 buffer_pool DEFAULT flash_cache DEFAULT cell_flash_cache DEFAULT) TABLESPACE "USERS" ENABLE,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         CONSTRAINT "FK_BANK"
   FOREIGN KEY ("BANK_ID") REFERENCES "BANK_REVIEWS"."BANKS" ("BANK_ID") ENABLE) SEGMENT creation IMMEDIATE PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS logging storage(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645 PCTINCREASE 0 FREELISTS 1 FREELIST groups 1 buffer_pool DEFAULT flash_cache DEFAULT cell_flash_cache DEFAULT) TABLESPACE "USERS" lob ("REVIEW_TEXT") store AS securefile (TABLESPACE "USERS" ENABLE
                                                                                                                                                                                                                                                                                                                                                                                                                             STORAGE IN ROW 4000 chunk 8192 NOCACHE logging NOCOMPRESS keep_duplicates storage(INITIAL 262144 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645 PCTINCREASE 0 buffer_pool DEFAULT flash_cache DEFAULT cell_flash_cache DEFAULT)) lob ("CLEANED_TEXT") store AS securefile (TABLESPACE "USERS" ENABLE
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         STORAGE IN ROW 4000 chunk 8192 NOCACHE logging NOCOMPRESS keep_duplicates storage(INITIAL 262144 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645 PCTINCREASE 0 buffer_pool DEFAULT flash_cache DEFAULT cell_flash_cache DEFAULT));


-- SEQUENCE definitions
