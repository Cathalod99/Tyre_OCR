import React, { useState } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  Alert,
  ScrollView,
  Dimensions,
} from 'react-native';
import { launchImageLibrary, launchCamera, ImagePickerResponse, MediaType } from 'react-native-image-picker';
import { request, PERMISSIONS, RESULTS } from 'react-native-permissions';
import Spinner from 'react-native-loading-spinner-overlay';
import Toast from 'react-native-toast-message';
import { TireOCRService } from './services/TireOCRService';

const { width, height } = Dimensions.get('window');

interface TireInfo {
  Manufacturer: string;
  'Tire model': string;
  'Tire size': string;
  'Load index and speed rating': string;
  'Scan TIN': {
    'Full DOT': string;
    'DOT Code': string;
    'Week Code': string;
    'Year Code': string;
    'Plant Code': string;
    'Plant Name': string;
    'Plant Country': string;
    'Plant Manufacturer': string;
  };
}

const App: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [tireInfo, setTireInfo] = useState<TireInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requestCameraPermission = async (): Promise<boolean> => {
    const result = await request(PERMISSIONS.IOS.CAMERA);
    return result === RESULTS.GRANTED;
  };

  const showImagePicker = () => {
    Alert.alert(
      'Select Image',
      'Choose an option',
      [
        { text: 'Camera', onPress: openCamera },
        { text: 'Photo Library', onPress: openImageLibrary },
        { text: 'Cancel', style: 'cancel' },
      ]
    );
  };

  const openCamera = async () => {
    const hasPermission = await requestCameraPermission();
    if (!hasPermission) {
      Alert.alert('Permission Denied', 'Camera permission is required to take photos');
      return;
    }

    const options = {
      mediaType: 'photo' as MediaType,
      quality: 0.8,
      maxWidth: 1024,
      maxHeight: 1024,
    };

    launchCamera(options, handleImagePickerResponse);
  };

  const openImageLibrary = () => {
    const options = {
      mediaType: 'photo' as MediaType,
      quality: 0.8,
      maxWidth: 1024,
      maxHeight: 1024,
    };

    launchImageLibrary(options, handleImagePickerResponse);
  };

  const handleImagePickerResponse = (response: ImagePickerResponse) => {
    if (response.didCancel || response.errorMessage) {
      return;
    }

    if (response.assets && response.assets[0]) {
      const asset = response.assets[0];
      if (asset.uri) {
        setSelectedImage(asset.uri);
        setTireInfo(null);
        setError(null);
      }
    }
  };

  const analyzeTire = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await TireOCRService.analyzeTire(selectedImage);
      setTireInfo(result);
      Toast.show({
        type: 'success',
        text1: 'Analysis Complete',
        text2: 'Tire information extracted successfully',
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to analyze tire';
      setError(errorMessage);
      Toast.show({
        type: 'error',
        text1: 'Analysis Failed',
        text2: errorMessage,
      });
    } finally {
      setLoading(false);
    }
  };

  const clearResults = () => {
    setSelectedImage(null);
    setTireInfo(null);
    setError(null);
  };

  const renderTireInfo = () => {
    if (!tireInfo) return null;

    return (
      <ScrollView style={styles.resultsContainer}>
        <Text style={styles.resultsTitle}>Tire Information</Text>
        
        <View style={styles.infoCard}>
          <Text style={styles.infoLabel}>Manufacturer:</Text>
          <Text style={styles.infoValue}>{tireInfo.Manufacturer || 'N/A'}</Text>
        </View>

        <View style={styles.infoCard}>
          <Text style={styles.infoLabel}>Model:</Text>
          <Text style={styles.infoValue}>{tireInfo['Tire model'] || 'N/A'}</Text>
        </View>

        <View style={styles.infoCard}>
          <Text style={styles.infoLabel}>Size:</Text>
          <Text style={styles.infoValue}>{tireInfo['Tire size'] || 'N/A'}</Text>
        </View>

        <View style={styles.infoCard}>
          <Text style={styles.infoLabel}>Load/Speed Rating:</Text>
          <Text style={styles.infoValue}>{tireInfo['Load index and speed rating'] || 'N/A'}</Text>
        </View>

        {tireInfo['Scan TIN'] && (
          <View style={styles.tinSection}>
            <Text style={styles.sectionTitle}>DOT/TIN Information</Text>
            
            <View style={styles.infoCard}>
              <Text style={styles.infoLabel}>Full DOT:</Text>
              <Text style={styles.infoValue}>{tireInfo['Scan TIN']['Full DOT'] || 'N/A'}</Text>
            </View>

            <View style={styles.infoCard}>
              <Text style={styles.infoLabel}>DOT Code:</Text>
              <Text style={styles.infoValue}>{tireInfo['Scan TIN']['DOT Code'] || 'N/A'}</Text>
            </View>

            <View style={styles.infoCard}>
              <Text style={styles.infoLabel}>Week/Year:</Text>
              <Text style={styles.infoValue}>
                {tireInfo['Scan TIN']['Week Code'] || 'N/A'}/{tireInfo['Scan TIN']['Year Code'] || 'N/A'}
              </Text>
            </View>

            {tireInfo['Scan TIN']['Plant Code'] && (
              <>
                <View style={styles.infoCard}>
                  <Text style={styles.infoLabel}>Plant Code:</Text>
                  <Text style={styles.infoValue}>{tireInfo['Scan TIN']['Plant Code']}</Text>
                </View>

                <View style={styles.infoCard}>
                  <Text style={styles.infoLabel}>Plant Name:</Text>
                  <Text style={styles.infoValue}>{tireInfo['Scan TIN']['Plant Name'] || 'N/A'}</Text>
                </View>

                <View style={styles.infoCard}>
                  <Text style={styles.infoLabel}>Country:</Text>
                  <Text style={styles.infoValue}>{tireInfo['Scan TIN']['Plant Country'] || 'N/A'}</Text>
                </View>
              </>
            )}
          </View>
        )}
      </ScrollView>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <Spinner visible={loading} textContent="Analyzing tire..." textStyle={styles.spinnerText} />
      
      <View style={styles.header}>
        <Text style={styles.title}>Tire OCR Scanner</Text>
        <Text style={styles.subtitle}>Extract tire information from photos</Text>
      </View>

      <ScrollView style={styles.content}>
        {selectedImage ? (
          <View style={styles.imageContainer}>
            <Image source={{ uri: selectedImage }} style={styles.image} />
            <View style={styles.buttonContainer}>
              <TouchableOpacity style={styles.analyzeButton} onPress={analyzeTire}>
                <Text style={styles.buttonText}>Analyze Tire</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.changeButton} onPress={showImagePicker}>
                <Text style={styles.buttonText}>Change Image</Text>
              </TouchableOpacity>
            </View>
          </View>
        ) : (
          <View style={styles.placeholderContainer}>
            <Text style={styles.placeholderText}>No image selected</Text>
            <TouchableOpacity style={styles.selectButton} onPress={showImagePicker}>
              <Text style={styles.buttonText}>Select Image</Text>
            </TouchableOpacity>
          </View>
        )}

        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {renderTireInfo()}

        {tireInfo && (
          <TouchableOpacity style={styles.clearButton} onPress={clearResults}>
            <Text style={styles.buttonText}>Clear Results</Text>
          </TouchableOpacity>
        )}
      </ScrollView>

      <Toast />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#2196F3',
    padding: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: 'white',
    opacity: 0.9,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  imageContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  image: {
    width: width - 40,
    height: 300,
    borderRadius: 10,
    marginBottom: 15,
  },
  placeholderContainer: {
    alignItems: 'center',
    padding: 40,
    backgroundColor: 'white',
    borderRadius: 10,
    marginBottom: 20,
  },
  placeholderText: {
    fontSize: 18,
    color: '#666',
    marginBottom: 20,
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: 10,
  },
  selectButton: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
  },
  analyzeButton: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
  },
  changeButton: {
    backgroundColor: '#FF9800',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
  },
  clearButton: {
    backgroundColor: '#F44336',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    alignSelf: 'center',
    marginTop: 20,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultsContainer: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    marginBottom: 20,
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  infoCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  infoLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#666',
    flex: 1,
  },
  infoValue: {
    fontSize: 16,
    color: '#333',
    flex: 2,
    textAlign: 'right',
  },
  tinSection: {
    marginTop: 20,
    paddingTop: 15,
    borderTopWidth: 2,
    borderTopColor: '#2196F3',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2196F3',
    marginBottom: 15,
    textAlign: 'center',
  },
  errorContainer: {
    backgroundColor: '#ffebee',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  errorText: {
    color: '#c62828',
    fontSize: 16,
    textAlign: 'center',
  },
  spinnerText: {
    color: '#fff',
  },
});

export default App;
