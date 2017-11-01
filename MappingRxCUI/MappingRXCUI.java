package MappingRXCUIandDrug;

import com.opencsv.CSVReader;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;


public class MappingRXCUI {
    public static void main(String[] args) throws IOException {
        MappingRXCUI mappingRXCUI = new MappingRXCUI();
        Map<String, String> RxCUIMap = mappingRXCUI.generateRXCUIMap("RxCUI.csv");
        mappingRXCUI.mappingRxCUIIntoFAERS("FAERS.csv", RxCUIMap);
    }

    public void mappingRxCUIIntoFAERS(String fileName, Map<String, String> map) throws IOException {
        BufferedReader br = null;
        BufferedWriter bw = null;
        try {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), StandardCharsets.UTF_8));
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("FAERSWithRxCUI.csv")));
            bw.write("dur,indications,lot_num,age_cod,occp_cod,reporter_country,auth_num,role_cod,to_mfr,outcomes,rept_dt,prod_ai,i_f_cod,age_grp,image,nda_num,dose_form,drugname,cum_dose_chr,primaryid,dechal,route,dur_cod,wt_cod,lit_ref,caseid,drug_seq,mfr_sndr,mfr_dt,rechal,event_dt,mfr_num,lot_nbr,dose_freq,e_sub,company,exp_dt,foll_seq,dose_vbm,wt,death_dt,end_dt,dose_amt,occr_country,caseversion,val_vbm,sex,dsg_drug_seq,gndr_cod,cum_dose_unit,dose_unit,rept_cod,reactions,age,quarter,i_f_code,RxCUI\n");
            String line;
            int count = 0;

            while ((line = br.readLine()) != null) {
                if (count == 0) {
                    count++;
                    continue;
                }

                line = line.trim();
                line = line.replaceAll("\\\\,", "");
                String[] parts = line.split(",");

                String activeIngredient = parts[11].trim();
                String doseForm = parts[16].trim();
                String tradeName = parts[17].trim();
                String strength = parts[43].trim();
                if (strength.length() != 0 && !Character.isDigit(strength.charAt(0))) {
                    strength = parts[42];
                }

                if (strength.length() != 0 && strength.contains(".")) {
                    strength = strength.substring(0, strength.indexOf("."));
                }

                String key1 = activeIngredient + " " +  strength + " " + doseForm;
                String key2 = activeIngredient + " " + strength;
                String key3 = activeIngredient;
                String key4 = tradeName;

                if (map.containsKey(key1)) {
                    bw.write(line + "," + map.get(key1));
                } else if (map.containsKey(key2)) {
                    bw.write(line + "," + map.get(key2));
                } else if (map.containsKey(key3)) {
                    bw.write(line + "," + map.get(key3));
                } else if (map.containsKey(key4)) {
                    bw.write(line + "," + map.get(key4));
                } else {
                    bw.write(line + "," + "");
                }

                bw.write("\n");

                count++;
            }
        }catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if(br!=null)
                br.close();
            if(bw!=null)
                bw.close();
        }
    }

    public Map<String, String> generateRXCUIMap(String fileName) throws IOException {
        Map<String, String> map = new HashMap<>();

        try {
            CSVReader reader = new CSVReader(new FileReader(fileName), ',');
            String line[];
            int count = 0;
            while ((line = reader.readNext()) != null) {
                if (count == 0) {
                    count++;
                    continue;
                }

                String activeIngredient = line[1].trim();
                String doseForm = line[2].trim();
                String tradeName = line[3].trim();
                String strength = line[5].trim();


                StringBuilder newStr = new StringBuilder();
                for (char c : strength.toCharArray()) {
                    if (Character.isDigit(c) || c == ' ') {
                        newStr.append(c);
                    }
                }
                strength = newStr.toString().trim();

                String RxCUI = line[20];
                map.put(activeIngredient + " " + strength + " " + doseForm, RxCUI);
                map.put(activeIngredient + " " + strength, RxCUI);
                map.put(activeIngredient, RxCUI);
                map.put(tradeName, RxCUI);

                count++;
            }
        }catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return map;
    }
}
