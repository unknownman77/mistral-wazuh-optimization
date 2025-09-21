# Untuk Log bisa dibuat sendiri dari internet, jadikan Log -> Rule

## Contoh
```bash
{"instruction": "Buat Wazuh rule untuk log berikut:\n2023-11-20T08:40:50.664842 WARNING ServiceA User96 192.168.1.102 Performance Warnings", "response": "<rule id=\"17102\" level=\"9\">\n    <if_group>authentication_success</if_group>\n    <weekday>weekends</weekday>\n    <description>Successful login during weekend.</description>\n    <mitre>\n      <id>T1078</id>\n    </mitre>\n    <group>login_day,pci_dss_10.2.5,pci_dss_10.6.1,gpg13_7.1,gpg13_7.2,gdpr_IV_35.7.d,gdpr_IV_32.2,hipaa_164.312.b,nist_800_53_AU.14,nist_800_53_AC.7,nist_800_53_AU.6,tsc_CC6.8,tsc_CC7.2,tsc_CC7.3,</group>\n  </rule>"}
```

## Nama File
```bash
logdata_wazuh_smart.jsonl
```
