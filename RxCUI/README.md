RxCUI Query Model
=======================================

### Steps to run:

1. Install maven: https://maven.apache.org/install.html
2. Put the data source (Drug master with levels) in rxcui/src/main/resources folder
3. `cd rxcui` folder, run `mvm clean install`
4. `cd rxcui/target`, run `java -jar rxcui-0.0.1-SNAPSHOT-jar-with-dependencies.jar drugMasterFileName`
5. The RxCUI query result for each drug is generated in rxcui/target named as `RxCUI_levels.csv`