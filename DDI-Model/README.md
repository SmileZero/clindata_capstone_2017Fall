DDI Model
=======================================

### Steps to run:

1. Install maven: https://maven.apache.org/install.html
2. Put the data source (Drug master with RxCUI) in ddi/src/main/resources folder
3. `cd ddi` folder, run `mvm clean install`
4. `cd ddi/target`, run `java -jar ddi-0.0.1-SNAPSHOT-jar-with-dependencies.jar dataSourceFileName`
5. Drugs with no interaction will display RxCUI in the console
6. The drug-drug interaction result is generated in ddi/target named as `DDI.csv`