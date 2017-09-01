from bs4 import BeautifulSoup, FeatureNotFound
import json
import xmltodict

class SECApi:

    def __init__(self, idx, type):
        self.idx = idx
        self.type = type
        self.logfile = 'sec.log';
        self.url = 'https://www.sec.gov/Archives/';
        
    def form4tojson(self,file):
        data = BeautifulSoup(file, 'html.parser').find('ownershipdocument');
        decom =  data.find('reportingowneraddress');
        if decom:
            decom.decompose();
        decom =  data.find('footnotes');
        if decom:
            decom.decompose();
        decom =  data.find('remarks');
        if decom:
            decom.decompose();
        decom =  data.find('ownersignature');
        if decom:
            decom.decompose();
        trunc = data.find('periodofreport');
        if trunc:
            onlydate = trunc.text[:10];
            trunc.string = onlydate;
        x=xmltodict.parse(data.prettify());
        j=json.dumps(x);
        return j;