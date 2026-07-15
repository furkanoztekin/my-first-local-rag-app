from foundry_local_sdk import FoundryLocalManager, Configuration

def main():
    print("Sistem kontrol ediliyor...")
    
    # Motoru başlat
    FoundryLocalManager.initialize(Configuration(app_name="debug_app"))
    catalog = FoundryLocalManager.instance.catalog
    
    print("\n--- Python'un Gördüğü Katalog Listesi ---")
    
    # Katalogdaki tüm modelleri tek tek yazdıralım
    # getattr ile 'models' veya 'list' gibi olası tüm özelliklere bakalım
    try:
        models = catalog.list_models() # Veya katalog bir liste/dict ise doğrudan erişim
        for model in models:
            print(f"Model ID: {model.id} | Alias: {model.alias}")
    except:
        # Eğer list_models() yoksa, katalog nesnesinin kendisini yazdıralım
        print(f"Katalog yapısı: {dir(catalog)}")
        print(f"Katalog içeriği: {catalog}")

if __name__ == '__main__':
    main()