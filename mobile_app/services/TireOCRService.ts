import axios, { AxiosResponse } from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Change this to your server URL

export interface TireInfo {
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

export class TireOCRService {
  static async analyzeTire(imageUri: string): Promise<TireInfo> {
    try {
      const formData = new FormData();
      formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'tire_image.jpg',
      } as any);

      const response: AxiosResponse<TireInfo> = await axios.post(
        `${API_BASE_URL}/analyze-tire`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 30000, // 30 seconds timeout
        }
      );

      if (response.status === 200) {
        return response.data;
      } else {
        throw new Error(`Server error: ${response.status}`);
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error('Cannot connect to server. Please check if the backend is running.');
        } else if (error.response) {
          throw new Error(`Server error: ${error.response.data?.detail || error.response.statusText}`);
        } else if (error.request) {
          throw new Error('Network error. Please check your internet connection.');
        }
      }
      throw new Error(error instanceof Error ? error.message : 'Unknown error occurred');
    }
  }

  static async healthCheck(): Promise<boolean> {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`, { timeout: 5000 });
      return response.status === 200;
    } catch {
      return false;
    }
  }
}
