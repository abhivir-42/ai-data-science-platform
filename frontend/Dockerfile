# Use Node.js 18 Alpine for smaller image size
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Add necessary packages for Next.js
RUN apk add --no-cache libc6-compat

# Install dependencies first for better caching
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy the application code
COPY . .

# Build the Next.js application
RUN npm run build

# Expose the port Next.js runs on
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Start the application
CMD ["npm", "run", "start"]
